"""Generate baseline ratings using the high-quality poem prompt."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from datetime import datetime
from main import BenchmarkArgs, format_pair_records, load_model_config, run_benchmark
from prompts import generation_good


BASELINE_INSTRUCTIONS = (
    "compose a polished, emotionally resonant piece with vivid imagery.",
    "ensure the poem has a deliberate structure and rhythmic flow.",
    "highlight precise language that develops a clear thematic arc.",
)


def parse_args() -> BenchmarkArgs:
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description="Run the good-poem baseline benchmark.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(f"baseline_results_{now}.json"),
        help="Where to write the JSON results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file listing models to evaluate per provider",
    )
    parser.add_argument(
        "--topics-limit",
        type=int,
        default=None,
        help="Restrict how many topics are used (useful for smoke tests)",
    )
    parser.add_argument(
        "--instructions-limit",
        type=int,
        default=None,
        help="Restrict how many baseline instruction variants are used",
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
        default=Path(f"baseline_pairs_{now}.json"),
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


def main() -> None:
    args = parse_args()
    model_config = load_model_config(args.config)
    results, pair_records = asyncio.run(
        run_benchmark(
            model_config,
            args,
            generation_fn=generation_good,
            instructions=BASELINE_INSTRUCTIONS,
        )
    )
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote baseline results to {args.output}")
    if args.pairs_output is not None:
        args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
        args.pairs_output.write_text(format_pair_records(pair_records), encoding="utf-8")
        print(f"Wrote poem and rating pairs to {args.pairs_output}")

if __name__ == "__main__":
    main()
