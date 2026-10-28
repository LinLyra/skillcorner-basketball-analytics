from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustness-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/robustness.png"))
    args = parser.parse_args()
    uncertainty = pd.read_csv(args.robustness_dir / "paired_bootstrap.csv")
    predictions = pd.read_csv(args.robustness_dir / "predictions.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    logloss = uncertainty[uncertainty["metric"] == "log_loss"].copy()
    logloss["label"] = logloss["defender_definition"].map(
        {
            "fixed_release": "Fixed release defender",
            "frame_nearest": "Frame-wise nearest",
            "frame_nearest_hysteresis_0_3m": "Nearest · 0.3 m hysteresis",
        }
    ) + " · " + logloss["quality_cohort"].map(
        {"all_eligible": "all eligible", "high_quality": "high quality"}
    )
    ax = axes[0]
    y = np.arange(len(logloss))
    low = logloss["mean_difference"] - logloss["ci95_low"]
    high = logloss["ci95_high"] - logloss["mean_difference"]
    ax.errorbar(
        logloss["mean_difference"], y, xerr=np.vstack([low, high]), fmt="o",
        color="#255F85", ecolor="#7098B4", capsize=4, linewidth=2,
    )
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, logloss["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Temporal − static log loss (95% cluster bootstrap CI)")
    ax.set_title("Every robustness interval includes zero")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1]
    primary = predictions[
        (predictions["defender_definition"] == "fixed_release")
        & (predictions["quality_cohort"] == "all_eligible")
    ]
    labels = {
        "M0_location_context": "Location + context",
        "M1_static_defense": "+ release distance",
        "M2_temporal_defense": "+ temporal defense",
    }
    colors = {"M0_location_context": "#6B7280", "M1_static_defense": "#2563EB", "M2_temporal_defense": "#E4572E"}
    for model, group in primary.groupby("model"):
        group = group.copy()
        group["bin"] = pd.qcut(group["probability"], q=6, duplicates="drop")
        calibration = group.groupby("bin", observed=True).agg(
            predicted=("probability", "mean"), observed=("outcome", "mean"), n=("outcome", "size")
        )
        ax.plot(calibration["predicted"], calibration["observed"], marker="o", linewidth=2,
                label=labels[model], color=colors[model])
    ax.plot([0, 1], [0, 1], linestyle="--", color="#111827", alpha=0.6)
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(0.15, 0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed make rate")
    ax.set_title("Out-of-game calibration · 1-second window")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Beyond the Snapshot — robustness and calibration", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
