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
    "M3_multiplayer_geometry": "+ multiplayer geometry",
    "M4_official_recalibrated": "Official SQ recalibrated",
    "M5_official_plus_temporal": "Official SQ + temporal",
    "Official_Shot_Quality_direct": "Official Shot Quality",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/figures/final_summary.png"))
    args = parser.parse_args()
    official = pd.read_csv("outputs/official_benchmark/scores.csv")
    spatial = pd.read_csv("outputs/spatial/scores.csv")
    robustness = pd.read_csv("outputs/robustness/paired_bootstrap.csv")

    models = [
        "M0_location_context", "M1_static_defense", "M2_temporal_defense",
        "M3_multiplayer_geometry", "Official_Shot_Quality_direct",
        "M4_official_recalibrated", "M5_official_plus_temporal",
    ]
    combined = pd.concat(
        [official[official["model"] != "M3_multiplayer_geometry"], spatial[spatial["model"] == "M3_multiplayer_geometry"]]
    )
    summary = combined.groupby("model")["log_loss"].agg(["mean", "std", "count"]).reindex(models)
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(summary))
    colors = ["#E4572E" if model == "Official_Shot_Quality_direct" else "#527C99" for model in summary.index]
    ax.barh(y, summary["mean"], xerr=summary["se"], color=colors, capsize=3)
    ax.set_yticks(y, [LABELS[model] for model in summary.index])
    ax.invert_yaxis()
    ax.set_xlim(0.59, 0.69)
    ax.set_xlabel("Mean held-out-game log loss (± SE)")
    ax.set_title("More features do not guarantee better generalization")
    ax.grid(axis="x", alpha=0.2)
    for position, (_, row) in enumerate(summary.iterrows()):
        ax.text(row["mean"] + row["se"] + 0.001, position, f"{row['mean']:.3f}", va="center", fontsize=9)

    ax = axes[1]
    robustness = robustness[
        (robustness["metric"] == "log_loss")
        & (robustness["model_a"] == "M2_temporal_defense")
    ].copy()
    robustness["label"] = robustness["defender_definition"].map(
        {"fixed_release": "Fixed defender", "frame_nearest": "Nearest each frame"}
    ) + " · " + robustness["quality_cohort"].map(
        {"all_eligible": "all", "high_quality": "high quality"}
    )
    y = np.arange(len(robustness))
    ax.errorbar(
        robustness["mean_difference"], y,
        xerr=np.vstack([
            robustness["mean_difference"] - robustness["ci95_low"],
            robustness["ci95_high"] - robustness["mean_difference"],
        ]),
        fmt="o", color="#255F85", ecolor="#7098B4", linewidth=2, capsize=4,
    )
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, robustness["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Temporal − static log loss (95% game-cluster CI)")
    ax.set_title("Temporal gain is small and not stable across games")
    ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Beyond the Snapshot — final evidence summary", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
