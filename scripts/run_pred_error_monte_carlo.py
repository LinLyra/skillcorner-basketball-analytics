from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skillcorner_basketball.features import build_shot_temporal_features
from skillcorner_basketball.io import load_dynamic_events, load_tracking
from skillcorner_basketball.modeling import leave_one_game_out_predictions, paired_game_bootstrap, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagate SkillCorner predError into M2-M1")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pred_error_mc"))
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    eligible = original.loc[
        (original["status"] == "ok") & (~original["fouled"])
        & original["release_distance_aligned_5cm"].eq(True), ["game_id", "shot_id"]
    ]
    eligible_keys = set(map(tuple, eligible.astype(str).to_numpy()))
    games = sorted(original["game_id"].astype(str).unique())
    cached = {}
    for game_id in games:
        directory = args.data_root / game_id
        print(f"Loading {game_id}", flush=True)
        _, entities = load_tracking(directory / f"{game_id}_tracking_data.jsonl.gz", game_id)
        shots = load_dynamic_events(directory / f"{game_id}_dynamic_events.json")["shots"]
        cached[game_id] = (shots, entities)

    rng = np.random.default_rng(args.seed)
    draw_rows, game_rows = [], []
    for draw in range(args.draws):
        tables = []
        for game_id, (shots, entities) in cached.items():
            table = build_shot_temporal_features(
                shots, entities, game_id=game_id, window_seconds=1.0,
                coordinate_noise_seed=int(rng.integers(0, 2**32 - 1)),
            )
            keys = list(zip(table["game_id"].astype(str), table["shot_id"].astype(str)))
            tables.append(table.loc[[key in eligible_keys for key in keys]])
        perturbed = pd.concat(tables, ignore_index=True)
        predictions = leave_one_game_out_predictions(perturbed)
        scores = score_predictions(predictions)
        pivot = scores.pivot(index="game_id", columns="model", values="log_loss")
        difference = pivot["M2_temporal_defense"] - pivot["M1_static_defense"]
        game_result = difference.rename("delta_log_loss").reset_index()
        game_result["draw"] = draw
        game_rows.append(game_result)
        bootstrap = paired_game_bootstrap(
            scores, metric="log_loss", model_a="M2_temporal_defense",
            model_b="M1_static_defense", repetitions=2_000, seed=args.seed + draw,
        )
        draw_rows.append({"draw": draw, **bootstrap})
        print(f"Monte Carlo draw {draw + 1}/{args.draws}", flush=True)

    draws = pd.DataFrame(draw_rows)
    pd.concat(game_rows, ignore_index=True).to_csv(args.output_dir / "game_differences.csv", index=False)
    draws.to_csv(args.output_dir / "draw_summary.csv", index=False)
    summary = {
        "draws": args.draws, "seed": args.seed, "shots_per_draw": len(eligible_keys),
        "assumption": "Independent isotropic Gaussian x/y errors; predError is the 90% radial coverage bound, so axis SD = predError/sqrt(-2*ln(0.1)).",
        "mean_delta_log_loss": float(draws["mean_difference"].mean()),
        "mc_2.5pct": float(draws["mean_difference"].quantile(0.025)),
        "mc_97.5pct": float(draws["mean_difference"].quantile(0.975)),
        "fraction_temporal_better": float((draws["mean_difference"] < 0).mean()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
