"""Poetry self-critique benchmark entry point."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime
from tqdm import tqdm
from prompts import discrimination, generation, specific_instructions, topics

try:  # Optional import: allows running with only one provider installed.
    from clients.openai_client import OpenAIClient, OpenAIConfig
except ImportError:  # pragma: no cover - keep running even if dependency missing
    OpenAIClient = None  # type: ignore[assignment]
    OpenAIConfig = None  # type: ignore[assignment]

try:
    from clients.anthropic_client import AnthropicClient, AnthropicConfig
except ImportError:  # pragma: no cover - keep running even if dependency missing
    AnthropicClient = None  # type: ignore[assignment]
    AnthropicConfig = None  # type: ignore[assignment]

load_dotenv()

DEFAULT_MODEL_CONFIG = {
    "openai": [
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-4.1-2025-04-14",
        "o3-2025-04-16",
    ],
    "anthropic": [
        "claude-3-7-sonnet-20250219",
        "claude-opus-4-20250514",
        "claude-opus-4-1-20250805",
        "claude-sonnet-4-20250514"
    ],
}


@dataclass
class BenchmarkArgs:
    output: Path
    config: Optional[Path]
    topics_limit: Optional[int]
    instructions_limit: Optional[int]
    max_concurrent_requests: int
    pairs_output: Optional[Path]


@dataclass
class PairRecord:
    provider: str
    model: str
    topic: str
    instruction: str
    poem: str
    rating_reply: str


def parse_args() -> BenchmarkArgs:
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description="Run the poetry self-critique benchmark.")
    parser.add_argument("--output", type=Path, default=Path(f"benchmark_results_{now}.json"), help="Where to write the JSON results")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON file listing models to evaluate per provider")
    parser.add_argument("--topics-limit", type=int, default=None, help="Restrict how many topics are used (useful for smoke tests)")
    parser.add_argument(
        "--instructions-limit",
        type=int,
        default=None,
        help="Restrict how many specific instruction variants are used",
    )
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=1,
        help="Maximum number of concurrent API requests (minimum 1)",
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path(f"benchmark_pairs_{now}.json"),
        help="Optional text file that records each generated poem next to its rating reply",
    )
    parsed = parser.parse_args()
    if parsed.max_concurrent_requests < 1:
        parser.error("--max-concurrent-requests must be at least 1")
    return BenchmarkArgs(
        output=parsed.output,
        config=parsed.config,
        topics_limit=parsed.topics_limit,
        instructions_limit=parsed.instructions_limit,
        max_concurrent_requests=parsed.max_concurrent_requests,
        pairs_output=parsed.pairs_output,
    )


def load_model_config(config_path: Optional[Path]) -> Dict[str, List[str]]:
    if config_path is None:
        return {provider: list(models) for provider, models in DEFAULT_MODEL_CONFIG.items()}
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Model configuration must map provider names to lists of model identifiers")
    parsed: Dict[str, List[str]] = {}
    for provider, entries in loaded.items():
        if not isinstance(entries, list):
            raise ValueError(f"Expected a list of model identifiers for provider '{provider}'")
        sanitized: List[str] = []
        for entry in entries:
            if isinstance(entry, str):
                sanitized.append(entry)
            elif isinstance(entry, dict) and "model" in entry:
                sanitized.append(str(entry["model"]))
            else:
                raise ValueError(
                    f"Provider '{provider}' entries must be strings or objects with a 'model' field"
                )
        parsed[provider] = sanitized
    return parsed


def build_openai_client() -> Optional[OpenAIClient]:
    if OpenAIClient is None or OpenAIConfig is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Skipping OpenAI provider: OPENAI_API_KEY is not set")
        return None
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAIClient(OpenAIConfig(api_key=api_key, base_url=base_url))


def build_anthropic_client() -> Optional[AnthropicClient]:
    if AnthropicClient is None or AnthropicConfig is None:
        return None
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Skipping Anthropic provider: ANTHROPIC_API_KEY is not set")
        return None
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    return AnthropicClient(AnthropicConfig(api_key=api_key, base_url=base_url))


def extract_rating(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    return max(1.0, min(10.0, value))


def select_subset(items: Iterable[str], limit: Optional[int]) -> List[str]:
    selected = list(items)
    if limit is not None:
        selected = selected[:limit]
    return selected


async def run_benchmark(
    model_config: Dict[str, List[str]],
    args: BenchmarkArgs,
    *,
    generation_fn: Callable[[str, str], str] = generation,
    instructions: Optional[Sequence[str]] = None,
    discrimination_fn: Callable[[str], str] = discrimination,
) -> Tuple[Dict[str, List[Dict[str, object]]], List[PairRecord]]:
    clients = {
        "openai": build_openai_client(),
        "anthropic": build_anthropic_client(),
    }

    chosen_topics = select_subset(topics, args.topics_limit)
    instruction_pool = list(instructions) if instructions is not None else list(specific_instructions)
    chosen_instructions = select_subset(instruction_pool, args.instructions_limit)

    results: Dict[str, List[Dict[str, object]]] = {}
    collected_pairs: List[PairRecord] = []

    for provider, model_names in model_config.items():
        client = clients.get(provider)
        if client is None:
            print(f"Skipping provider '{provider}' because no API client is available")
            continue

        provider_results: List[Dict[str, object]] = []
        for model_name in model_names:
            ratings: List[float] = []
            failures = 0
            semaphore = asyncio.Semaphore(args.max_concurrent_requests)

            async def call_with_limit(func: Callable[..., str], *call_args: object) -> str:
                async with semaphore:
                    return await asyncio.to_thread(func, *call_args)

            async def process_pair(topic: str, instruction: str) -> Tuple[Optional[float], int, Optional[PairRecord]]:
                generation_prompt = generation_fn(topic, instruction)
                print(f"[{provider}] generating with model: {model_name}")
                try:
                    poem = await call_with_limit(client.generate_text, generation_prompt, model_name)
                except Exception as exc:  # pragma: no cover - surface API issues without crashing everything
                    print(f"[{provider}] Failed to generate with {model_name}: {exc}")
                    return None, 1, None

                rating_prompt = discrimination_fn(poem)
                try:
                    rating_reply = await call_with_limit(client.rate_text, rating_prompt, model_name)
                except Exception as exc:  # pragma: no cover - surface API issues without crashing everything
                    print(f"[{provider}] Failed to rate with {model_name}: {exc}")
                    return None, 1, None

                pair_record = PairRecord(
                    provider=provider,
                    model=model_name,
                    topic=topic,
                    instruction=instruction,
                    poem=poem,
                    rating_reply=rating_reply,
                )

                score = extract_rating(rating_reply)
                if score is None:
                    print(f"[{provider}] Could not parse rating from '{rating_reply}'")
                    return None, 1, pair_record

                return score, 0, pair_record

            tasks = [
                asyncio.create_task(process_pair(topic, instruction))
                for topic in chosen_topics
                for instruction in chosen_instructions
            ]

            total_tasks = len(tasks)
            progress = tqdm(total=total_tasks, desc=f"{provider}:{model_name}", leave=False)
            try:
                if tasks:
                    for task in asyncio.as_completed(tasks):
                        score, failure_increment, pair_record = await task
                        if score is not None:
                            ratings.append(score)
                        failures += failure_increment
                        if pair_record is not None:
                            collected_pairs.append(pair_record)
                        progress.update(1)
                else:
                    progress.total = 0
                    progress.refresh()
            finally:
                progress.close()

            average = statistics.mean(ratings) if ratings else None
            provider_results.append(
                {
                    "model": model_name,
                    "average_rating": average,
                    "ratings": ratings,
                    "samples": len(ratings),
                    "failures": failures,
                }
            )

        results[provider] = provider_results

    return results, collected_pairs


def format_pair_records(pairs: Sequence[PairRecord]) -> str:
    if not pairs:
        return "No successful poem-rating pairs were recorded.\n"

    total = len(pairs)
    lines: List[str] = []
    for index, record in enumerate(pairs, start=1):
        lines.append(f"[{index}] Provider: {record.provider} | Model: {record.model}")
        lines.append(f"Topic: {record.topic}")
        lines.append(f"Instruction: {record.instruction}")
        lines.append("Poem:")
        lines.append(record.poem.rstrip("\n"))
        lines.append("")
        lines.append("Rating reply:")
        lines.append(record.rating_reply.rstrip("\n"))
        if index < total:
            lines.append("-" * 80)
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    model_config = load_model_config(args.config)
    results, pair_records = asyncio.run(run_benchmark(model_config, args))
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results to {args.output}")

    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    args.pairs_output.write_text(format_pair_records(pair_records), encoding="utf-8")
    print(f"Wrote poem and rating pairs to {args.pairs_output}")

if __name__ == "__main__":
    main()
