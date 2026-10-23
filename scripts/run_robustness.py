from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skillcorner_basketball.features import build_shot_temporal_features
from skillcorner_basketball.io import load_dynamic_events, load_tracking
from skillcorner_basketball.modeling import (
    leave_one_game_out_predictions,
    paired_game_bootstrap,
    score_predictions,
)


def eligible(data: pd.DataFrame, *, high_quality: bool) -> pd.DataFrame:
    mask = (
        (data["status"] == "ok")
        & (~data["fouled"])
        & data["release_distance_aligned_5cm"].eq(True)
    )
    if high_quality:
        mask &= (data["both_detected_rate"] >= 0.8) & (
            data["mean_combined_pred_error_m"] <= 1.0
        )
    return data[mask].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run defender-definition and quality robustness")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robustness"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    nearest_tables = []
    games = sorted(fixed["game_id"].astype(str).unique())
    for game_id in games:
        directory = args.data_root / game_id
        print(f"Nearest-defender features: {game_id}", flush=True)
        _, entities = load_tracking(directory / f"{game_id}_tracking_data.jsonl.gz", game_id)
        shots = load_dynamic_events(directory / f"{game_id}_dynamic_events.json")["shots"]
        nearest_tables.append(
            build_shot_temporal_features(
                shots,
                entities,
                game_id=game_id,
                window_seconds=1.0,
                defender_mode="frame_nearest",
            )
        )
    nearest = pd.concat(nearest_tables, ignore_index=True)
    nearest.to_csv(args.output_dir / "shot_features_frame_nearest_1s.csv", index=False)

    score_tables = []
    prediction_tables = []
    uncertainty = []
    cohort_rows = []
    for definition, data in (("fixed_release", fixed), ("frame_nearest", nearest)):
        for quality in ("all_eligible", "high_quality"):
            cohort = eligible(data, high_quality=quality == "high_quality")
            predictions = leave_one_game_out_predictions(cohort)
            predictions["defender_definition"] = definition
            predictions["quality_cohort"] = quality
            scores = score_predictions(predictions)
            scores["defender_definition"] = definition
            scores["quality_cohort"] = quality
            prediction_tables.append(predictions)
            score_tables.append(scores)
            cohort_rows.append(
                {
                    "defender_definition": definition,
                    "quality_cohort": quality,
                    "shots": len(cohort),
                    "games": cohort["game_id"].nunique(),
                    "made_rate": cohort["outcome"].mean(),
                }
            )
            for metric in ("log_loss", "brier", "auc"):
                result = paired_game_bootstrap(
                    scores,
                    metric=metric,
                    model_a="M2_temporal_defense",
                    model_b="M1_static_defense",
                )
                result.update(
                    {"defender_definition": definition, "quality_cohort": quality}
                )
                uncertainty.append(result)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    scores = pd.concat(score_tables, ignore_index=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(cohort_rows).to_csv(args.output_dir / "cohorts.csv", index=False)

    primary = predictions[
        (predictions["defender_definition"] == "fixed_release")
        & (predictions["quality_cohort"] == "all_eligible")
    ].pivot(index=["game_id", "shot_id", "outcome"], columns="model", values="probability")
    primary["temporal_minus_static_probability"] = (
        primary["M2_temporal_defense"] - primary["M1_static_defense"]
    )
    primary["absolute_disagreement"] = primary["temporal_minus_static_probability"].abs()
    disagreements = primary.reset_index().sort_values("absolute_disagreement", ascending=False)
    disagreements.head(50).to_csv(args.output_dir / "top_model_disagreements.csv", index=False)

    manifest = {
        "window_seconds": 1.0,
        "defender_definitions": ["fixed_release", "frame_nearest"],
        "high_quality_rule": "both_detected_rate >= 0.8 and mean_combined_pred_error_m <= 1.0",
        "bootstrap": "20,000 paired cluster resamples of held-out games; seed 20261024",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(uncertainty).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
