"""Visualize benchmark results with matplotlib bar charts."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark averages with error bars.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark_results_20250925_172102.json"),
        help="Path to the benchmark results JSON file",
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
    data: Dict[str, List[Dict[str, object]]]
) -> List[Tuple[str, str, float, float, int]]:
    """Flatten the benchmark results into plot-friendly tuples.

    Each tuple contains (provider, model, discernment_mean, discernment_std, sample_count).
    Discernment score is defined as the absolute distance of each rating from 10.
    Standard deviation falls back to zero when fewer than two ratings are available.
    """

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
                # Skip models without an average; they likely failed entirely.
                continue
            if not isinstance(ratings, Iterable):
                raise ValueError(f"Provider '{provider}' model '{model}' ratings must be iterable")
            ratings_list = [float(value) for value in ratings]
            distances = [abs(10.0 - value) for value in ratings_list]
            if len(distances) >= 2:
                std_dev = statistics.pstdev(distances)
            else:
                std_dev = 0.0
            mean_distance = statistics.fmean(distances) if distances else 0.0
            series.append((provider, model, mean_distance, std_dev, len(ratings_list)))
    return series


def plot_results(series: List[Tuple[str, str, float, float, int]]) -> plt.Figure:
    if not series:
        raise ValueError("No plot data available; ensure the results include averages")

    sorted_series = sorted(series, key=lambda item: item[2], reverse=True)

    providers = sorted({row[0] for row in sorted_series})
    cmap = plt.get_cmap("tab10")

    labels = [f"{model}" for provider, model, *_ in sorted_series]
    scores = [row[2] for row in sorted_series]
    std_devs = [row[3] for row in sorted_series]
    sample_counts = [row[4] for row in sorted_series]

    fig, ax = plt.subplots(figsize=(12, max(4, 0.6 * len(labels))))
    positions = range(len(labels))
    ax.bar(positions, scores, yerr=std_devs, capsize=6)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 10)
    ax.set_ylabel("Discernment score")
    ax.set_title("inverse rating averages")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for index, (score, sample_count) in enumerate(zip(scores, sample_counts)):
        ax.text(
            index,
            score + 0.15,
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
    series = compute_series(data)
    fig = plot_results(series)

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved chart to {args.output}")
    if args.show or not args.output:
        plt.show()


if __name__ == "__main__":
    main()
