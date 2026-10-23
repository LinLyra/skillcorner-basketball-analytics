from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from skillcorner_basketball.features import build_shot_temporal_features
from skillcorner_basketball.io import load_dynamic_events, load_tracking
from skillcorner_basketball.modeling import leave_one_game_out_predictions, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered temporal-context study")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--windows", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games = sorted(
        directory.name
        for directory in args.data_root.iterdir()
        if directory.is_dir()
        and (directory / f"{directory.name}_tracking_data.jsonl.gz").exists()
    )
    feature_tables: dict[float, list[pd.DataFrame]] = {window: [] for window in args.windows}
    audit_rows = []
    for game_id in games:
        directory = args.data_root / game_id
        print(f"Loading game {game_id}...", flush=True)
        _, entities = load_tracking(directory / f"{game_id}_tracking_data.jsonl.gz", game_id)
        shots = load_dynamic_events(directory / f"{game_id}_dynamic_events.json")["shots"]
        for window in args.windows:
            features = build_shot_temporal_features(
                shots, entities, game_id=game_id, window_seconds=window
            )
            feature_tables[window].append(features)
            audit_rows.append(
                {
                    "game_id": game_id,
                    "window_seconds": window,
                    "shots": len(features),
                    "valid": int((features["status"] == "ok").sum()),
                    "aligned_5cm": int(
                        features["release_distance_aligned_5cm"].eq(True).sum()
                    ),
                    "median_detected_rate": features["both_detected_rate"].median(),
                    "median_pred_error_m": features["mean_combined_pred_error_m"].median(),
                }
            )
        del entities

    all_scores = []
    for window, tables in feature_tables.items():
        data = pd.concat(tables, ignore_index=True)
        data.to_csv(args.output_dir / f"shot_features_{window:g}s.csv", index=False)
        eligible = data[
            (data["status"] == "ok")
            & (~data["fouled"])
            & (data["release_distance_aligned_5cm"] == True)  # noqa: E712
        ].copy()
        predictions = leave_one_game_out_predictions(eligible)
        predictions["window_seconds"] = window
        predictions.to_csv(args.output_dir / f"predictions_{window:g}s.csv", index=False)
        scores = score_predictions(predictions)
        scores["window_seconds"] = window
        all_scores.append(scores)

    audit = pd.DataFrame(audit_rows)
    scores = pd.concat(all_scores, ignore_index=True)
    audit.to_csv(args.output_dir / "data_audit.csv", index=False)
    scores.to_csv(args.output_dir / "logo_scores.csv", index=False)
    summary = (
        scores.groupby(["window_seconds", "model"])[["log_loss", "brier", "auc", "calibration_in_large"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    manifest = {
        "games": games,
        "windows_seconds": args.windows,
        "eligibility": "status=ok, non-fouled, official/recomputed release distance within 5cm",
        "validation": "leave-one-game-out; fixed logistic-regression specifications",
    }
    (args.output_dir / "study_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
