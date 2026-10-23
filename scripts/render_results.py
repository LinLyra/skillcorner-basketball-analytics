from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "M0_location_context": "Location + context",
    "M1_static_defense": "+ release distance",
    "M2_temporal_defense": "+ temporal defense",
}
COLORS = {"M0_location_context": "#6B7280", "M1_static_defense": "#2563EB", "M2_temporal_defense": "#E4572E"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/preliminary_results.png"))
    args = parser.parse_args()
    scores = pd.read_csv(args.study_dir / "logo_scores.csv")
    features = pd.read_csv(args.study_dir / "shot_features_1s.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    ax = axes[0]
    summary = scores.groupby(["window_seconds", "model"])["log_loss"].agg(["mean", "std"])
    for model in LABELS:
        subset = summary.xs(model, level="model")
        ax.errorbar(
            subset.index,
            subset["mean"],
            yerr=subset["std"] / np.sqrt(10),
            marker="o",
            linewidth=2,
            capsize=3,
            label=LABELS[model],
            color=COLORS[model],
        )
    ax.set_title("Out-of-game prediction by temporal window")
    ax.set_xlabel("History window (seconds)")
    ax.set_ylabel("Log loss (lower is better)")
    ax.set_xticks([0.5, 1.0, 2.0])
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    one_second = scores[scores["window_seconds"] == 1].pivot(
        index="game_id", columns="model", values="log_loss"
    )
    delta = one_second["M2_temporal_defense"] - one_second["M1_static_defense"]
    colors = np.where(delta < 0, "#2A9D8F", "#D1495B")
    positions = np.arange(len(delta))
    ax.barh(positions, delta, color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(positions, [str(x) for x in delta.index])
    ax.set_xlabel("Temporal − static log loss")
    ax.set_ylabel("Held-out game")
    ax.set_title("Temporal context helps inconsistently")
    ax.grid(axis="x", alpha=0.2)

    eligible = features[
        (features["status"] == "ok")
        & (~features["fouled"])
        & (features["release_distance_aligned_5cm"] == True)  # noqa: E712
    ]
    fig.suptitle(
        f"Beyond the Snapshot — preliminary fixed-specification results\n"
        f"10 games · {len(features):,} shots · {len(eligible):,} eligible · leave-one-game-out",
        fontsize=14,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
