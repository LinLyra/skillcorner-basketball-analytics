from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from skillcorner_basketball.io import load_dynamic_events, load_tracking
from skillcorner_basketball.modeling import (
    MODEL_SPECS,
    SPATIAL_MODEL_SPEC,
    leave_one_game_out_predictions,
    paired_game_bootstrap,
    score_predictions,
)
from skillcorner_basketball.spatial import build_shot_spatial_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiplayer spatial geometry")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spatial"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    tables = []
    for game_id in sorted(temporal["game_id"].astype(str).unique()):
        directory = args.data_root / game_id
        print(f"Spatial geometry: {game_id}", flush=True)
        _, entities = load_tracking(directory / f"{game_id}_tracking_data.jsonl.gz", game_id)
        shots = load_dynamic_events(directory / f"{game_id}_dynamic_events.json")["shots"]
        tables.append(build_shot_spatial_features(shots, entities, game_id=game_id))
    spatial = pd.concat(tables, ignore_index=True)
    spatial["game_id"] = spatial["game_id"].astype(int)
    spatial.to_csv(args.output_dir / "spatial_features.csv", index=False)
    data = temporal.merge(spatial, on=["game_id", "shot_id"], how="left")
    eligible = data[
        (data["status"] == "ok")
        & (data["spatial_status"] == "ok")
        & (~data["fouled"])
        & data["release_distance_aligned_5cm"].eq(True)
    ].copy()
    specs = MODEL_SPECS + (SPATIAL_MODEL_SPEC,)
    predictions = leave_one_game_out_predictions(eligible, specs=specs)
    scores = score_predictions(predictions)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    comparisons = []
    for baseline in ("M1_static_defense", "M2_temporal_defense"):
        for metric in ("log_loss", "brier", "auc"):
            result = paired_game_bootstrap(
                scores,
                metric=metric,
                model_a="M3_multiplayer_geometry",
                model_b=baseline,
            )
            comparisons.append(result)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    summary = scores.groupby("model")[["log_loss", "brier", "auc"]].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "summary.csv")
    print(f"Eligible shots: {len(eligible)}", flush=True)
    print(summary.to_string(), flush=True)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
