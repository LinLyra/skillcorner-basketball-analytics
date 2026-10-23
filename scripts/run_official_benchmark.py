from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from skillcorner_basketball.modeling import (
    MODEL_SPECS,
    ModelSpec,
    leave_one_game_out_predictions,
    paired_game_bootstrap,
    score_predictions,
)


OFFICIAL_CALIBRATED = ModelSpec(
    "M4_official_recalibrated", ("official_shot_quality",), categorical=()
)
OFFICIAL_PLUS_TEMPORAL = ModelSpec(
    "M5_official_plus_temporal",
    (
        "official_shot_quality",
        "recent_closing_speed_mps",
        "separation_gain_m",
        "pressure_duration_s",
        "mean_distance_m",
        "min_distance_m",
        "max_distance_m",
    ),
    categorical=(),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark against official SkillCorner Shot Quality")
    parser.add_argument("--features", type=Path, default=Path("outputs/study/shot_features_1s.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/official_benchmark"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.features)
    eligible = data[
        (data["status"] == "ok")
        & (~data["fouled"])
        & data["release_distance_aligned_5cm"].eq(True)
        & data["official_shot_quality"].notna()
    ].copy()

    specs = MODEL_SPECS + (OFFICIAL_CALIBRATED, OFFICIAL_PLUS_TEMPORAL)
    predictions = leave_one_game_out_predictions(eligible, specs=specs)
    direct = eligible[["game_id", "shot_id", "outcome"]].copy()
    direct["model"] = "Official_Shot_Quality_direct"
    direct["probability"] = np.clip(eligible["official_shot_quality"] / 100, 1e-6, 1 - 1e-6)
    predictions = pd.concat([predictions, direct], ignore_index=True)
    scores = score_predictions(predictions)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    summary = scores.groupby("model")[["log_loss", "brier", "auc"]].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "summary.csv")

    comparisons = []
    for model_a, model_b in (
        ("M5_official_plus_temporal", "M4_official_recalibrated"),
        ("M5_official_plus_temporal", "Official_Shot_Quality_direct"),
        ("M2_temporal_defense", "Official_Shot_Quality_direct"),
    ):
        for metric in ("log_loss", "brier", "auc"):
            comparison = paired_game_bootstrap(
                scores, metric=metric, model_a=model_a, model_b=model_b
            )
            comparisons.append(comparison)
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    print(summary.to_string(), flush=True)
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
