from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from skillcorner_basketball.features import build_shot_temporal_features
from skillcorner_basketball.io import load_dynamic_events, load_tracking
from skillcorner_basketball.modeling import (
    leave_one_game_out_predictions,
    paired_game_bootstrap,
    score_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trajectory smoothing sensitivity")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoothing"))
    parser.add_argument("--smoothing-frames", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    games = sorted(path.name for path in args.data_root.iterdir() if path.is_dir())
    for game_id in games:
        directory = args.data_root / game_id
        tracking_path = directory / f"{game_id}_tracking_data.jsonl.gz"
        if not tracking_path.exists():
            continue
        print(f"Smoothed features: {game_id}", flush=True)
        _, entities = load_tracking(tracking_path, game_id)
        shots = load_dynamic_events(directory / f"{game_id}_dynamic_events.json")["shots"]
        tables.append(
            build_shot_temporal_features(
                shots,
                entities,
                game_id=game_id,
                window_seconds=1.0,
                defender_mode="fixed_release",
                smoothing_frames=args.smoothing_frames,
            )
        )
    data = pd.concat(tables, ignore_index=True)
    data.to_csv(args.output_dir / "shot_features_smoothed_1s.csv", index=False)
    eligible = data[
        (data["status"] == "ok")
        & (~data["fouled"])
        & data["release_distance_aligned_5cm"].eq(True)
    ].copy()
    predictions = leave_one_game_out_predictions(eligible)
    scores = score_predictions(predictions)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    rows = []
    for metric in ("log_loss", "brier", "auc"):
        rows.append(
            paired_game_bootstrap(
                scores,
                metric=metric,
                model_a="M2_temporal_defense",
                model_b="M1_static_defense",
            )
        )
    result = pd.DataFrame(rows)
    result["smoothing_frames"] = args.smoothing_frames
    result.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
