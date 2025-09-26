"""Visualize benchmark results against a good-poem baseline."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline ratios with error bars.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_results_20250925_184054.json"),
        help="Path to the benchmark (bad poem) results JSON file",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("baseline_results.json"),
        help="Path to the good-poem baseline results JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the generated figure",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving",
    )
    return parser.parse_args()


def load_results(path: Path) -> Dict[str, List[Dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Benchmark results JSON must map providers to model entries")
    return data


def compute_series(
    data: Dict[str, List[Dict[str, object]]],
    baseline_data: Dict[str, List[Dict[str, object]]],
) -> List[Tuple[str, str, float, float, int]]:
    """Return (provider, model, ratio_mean, ratio_std, sample_count) tuples.

    Ratio per sample is defined as baseline_avg / rating_value, using the baseline
    average rating for the corresponding provider/model.
    """

    baseline_index: Dict[str, Dict[str, Dict[str, object]]] = {}
    for provider, entries in baseline_data.items():
        if not isinstance(entries, Iterable):
            raise ValueError(f"Baseline provider '{provider}' entries must be iterable")
        provider_map: Dict[str, Dict[str, object]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Baseline provider '{provider}' entry must be an object")
            model_name = str(entry.get("model", "unknown"))
            provider_map[model_name] = entry
        baseline_index[provider] = provider_map

    series: List[Tuple[str, str, float, float, int]] = []
    for provider, entries in data.items():
        if not isinstance(entries, Iterable):
            raise ValueError(f"Provider '{provider}' entries must be iterable")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Provider '{provider}' entry must be an object")
            model = str(entry.get("model", "unknown"))
            average = entry.get("average_rating")
            ratings = entry.get("ratings", [])
            if average is None:
                continue
            if not isinstance(ratings, Iterable):
                raise ValueError(f"Provider '{provider}' model '{model}' ratings must be iterable")

            baseline_entry = baseline_index.get(provider, {}).get(model)
            if not baseline_entry:
                print(f"Warning: missing baseline data for {provider}:{model}; skipping")
                continue
            baseline_average = baseline_entry.get("average_rating")
            if baseline_average is None:
                print(f"Warning: baseline average missing for {provider}:{model}; skipping")
                continue

            ratings_list = [float(value) for value in ratings]
            baseline_avg_value = float(baseline_average)
            ratios = [baseline_avg_value / value for value in ratings_list if value]
            if not ratios:
                continue
            mean_ratio = statistics.fmean(ratios)
            std_dev = statistics.pstdev(ratios) if len(ratios) >= 2 else 0.0
            series.append((provider, model, mean_ratio, std_dev, len(ratios)))
    return series


def plot_results(series: List[Tuple[str, str, float, float, int]]) -> plt.Figure:
    if not series:
        raise ValueError("No plot data available; ensure the results include averages")

    sorted_series = sorted(series, key=lambda item: item[2], reverse=True)


    labels = [f"{model}" for provider, model, *_ in sorted_series]
    ratios = [row[2] for row in sorted_series]
    std_devs = [row[3] for row in sorted_series]
    sample_counts = [row[4] for row in sorted_series]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(labels))))
    positions = range(len(labels))
    ax.bar(positions, ratios, yerr=std_devs, capsize=6)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    max_ratio = max(ratios)
    ax.set_ylim(0, max(1.0, max_ratio * 1.15))
    ax.set_ylabel("Baseline ratio (good / benchmark)")
    ax.set_title("Model baseline comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    offset = max(0.05, max_ratio * 0.03)
    for index, (score, sample_count) in enumerate(zip(ratios, sample_counts)):
        ax.text(
            index,
            score + offset,
            f"n={sample_count}",
            ha="center",
            fontsize=9,
            color="#555555",
        )

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    data = load_results(args.input)
    baseline_data = load_results(args.baseline)
    series = compute_series(data, baseline_data)
    fig = plot_results(series)

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved chart to {args.output}")
    if args.show or not args.output:
        plt.show()


if __name__ == "__main__":
    main()
