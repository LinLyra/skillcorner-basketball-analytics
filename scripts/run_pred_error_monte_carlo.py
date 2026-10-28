from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skillcorner_basketball.io import load_tracking
from skillcorner_basketball.modeling import leave_one_game_out_predictions, score_predictions


def normalise_id(value) -> str:
    return str(int(float(value)))


def prepare_trajectories(data: pd.DataFrame, data_root: Path) -> dict[str, np.ndarray]:
    """Extract the fixed-defender rows once; Monte Carlo then stays vectorized."""
    pieces = []
    for game_id, shots in data.groupby(data["game_id"].astype(str), sort=True):
        directory = data_root / game_id
        print(f"Loading {game_id}", flush=True)
        _, entities = load_tracking(directory / f"{game_id}_tracking_data.jsonl.gz", game_id)
        players = entities[entities["entity_type"] == "player"].copy()
        players["player_key"] = players["player_id"].map(normalise_id)
        by_player = {
            key: group.set_index("frame_idx")
            for key, group in players.groupby("player_key", sort=False)
        }
        for shot_index, shot in shots.iterrows():
            release = int(shot["release_frame"])
            shooter = by_player[normalise_id(shot["shooter_id"])]
            defender = by_player[normalise_id(shot["defender_id"])]
            columns = ["x_m", "y_m", "pred_error_m"]
            trajectory = shooter.loc[shooter.index.to_series().between(release - 25, release), columns].join(
                defender.loc[defender.index.to_series().between(release - 25, release), columns],
                how="inner", lsuffix="_shooter", rsuffix="_defender",
            ).dropna(subset=["x_m_shooter", "y_m_shooter", "x_m_defender", "y_m_defender"])
            trajectory = trajectory.sort_index().reset_index()
            trajectory["shot_code"] = shot_index
            trajectory["seconds_to_release"] = (trajectory["frame_idx"] - release) / 25.0
            pieces.append(trajectory)
        del entities, players, by_player
    long = pd.concat(pieces, ignore_index=True).sort_values(["shot_code", "frame_idx"])
    codes = long["shot_code"].to_numpy(dtype=int)
    counts = np.bincount(codes, minlength=len(data))
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    ends = np.cumsum(counts)
    if np.any(counts < 2):
        raise ValueError("Every Monte Carlo trajectory must contain at least two frames")
    radius_to_axis_sd = np.sqrt(-2.0 * np.log(0.1))
    return {
        "codes": codes,
        "counts": counts.astype(float),
        "starts": starts,
        "ends": ends,
        "t": long["seconds_to_release"].to_numpy(dtype=float),
        "sx": long["x_m_shooter"].to_numpy(dtype=float),
        "sy": long["y_m_shooter"].to_numpy(dtype=float),
        "dx": long["x_m_defender"].to_numpy(dtype=float),
        "dy": long["y_m_defender"].to_numpy(dtype=float),
        "ssd": long["pred_error_m_shooter"].fillna(0).clip(lower=0).to_numpy(dtype=float) / radius_to_axis_sd,
        "dsd": long["pred_error_m_defender"].fillna(0).clip(lower=0).to_numpy(dtype=float) / radius_to_axis_sd,
    }


def perturb_features(base: pd.DataFrame, trajectory: dict[str, np.ndarray], rng: np.random.Generator) -> pd.DataFrame:
    codes = trajectory["codes"]
    n_shots = len(base)
    sx = trajectory["sx"] + rng.normal(0, trajectory["ssd"])
    sy = trajectory["sy"] + rng.normal(0, trajectory["ssd"])
    dx = trajectory["dx"] + rng.normal(0, trajectory["dsd"])
    dy = trajectory["dy"] + rng.normal(0, trajectory["dsd"])
    distance = np.hypot(dx - sx, dy - sy)
    counts = trajectory["counts"]
    sum_distance = np.bincount(codes, weights=distance, minlength=n_shots)
    minimum = np.minimum.reduceat(distance, trajectory["starts"])
    maximum = np.maximum.reduceat(distance, trajectory["starts"])
    start_distance = distance[trajectory["starts"]]
    release_distance = distance[trajectory["ends"] - 1]

    t = trajectory["t"]
    sum_t = np.bincount(codes, weights=t, minlength=n_shots)
    sum_t2 = np.bincount(codes, weights=t * t, minlength=n_shots)
    sum_td = np.bincount(codes, weights=t * distance, minlength=n_shots)
    denominator = sum_t2 - sum_t * sum_t / counts
    slope = (sum_td - sum_t * sum_distance / counts) / denominator

    recent = t >= -0.4 - 1e-12
    recent_codes = codes[recent]
    recent_t = t[recent]
    recent_d = distance[recent]
    recent_n = np.bincount(recent_codes, minlength=n_shots).astype(float)
    recent_sum_t = np.bincount(recent_codes, weights=recent_t, minlength=n_shots)
    recent_sum_t2 = np.bincount(recent_codes, weights=recent_t * recent_t, minlength=n_shots)
    recent_sum_d = np.bincount(recent_codes, weights=recent_d, minlength=n_shots)
    recent_sum_td = np.bincount(recent_codes, weights=recent_t * recent_d, minlength=n_shots)
    recent_denominator = recent_sum_t2 - recent_sum_t * recent_sum_t / recent_n
    recent_slope = (recent_sum_td - recent_sum_t * recent_sum_d / recent_n) / recent_denominator

    result = base.copy()
    result["release_distance_m"] = release_distance
    result["window_start_distance_m"] = start_distance
    result["mean_distance_m"] = sum_distance / counts
    result["min_distance_m"] = minimum
    result["max_distance_m"] = maximum
    result["separation_gain_m"] = release_distance - start_distance
    result["net_closing_speed_mps"] = -slope
    result["recent_closing_speed_mps"] = -recent_slope
    result["pressure_duration_s"] = np.bincount(
        codes, weights=(distance <= 1.5), minlength=n_shots
    ) / 25.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagate SkillCorner predError into M2-M1")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pred_error_mc"))
    parser.add_argument("--report-summary", type=Path, default=Path("reports/tables/pred_error_mc_summary.json"))
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    data = original.loc[
        (original["status"] == "ok") & (~original["fouled"])
        & original["release_distance_aligned_5cm"].eq(True)
    ].reset_index(drop=True)
    trajectory = prepare_trajectories(data, args.data_root)
    rng = np.random.default_rng(args.seed)
    draw_rows, game_rows = [], []
    for draw in range(args.draws):
        perturbed = perturb_features(data, trajectory, rng)
        predictions = leave_one_game_out_predictions(perturbed)
        scores = score_predictions(predictions)
        pivot = scores.pivot(index="game_id", columns="model", values="log_loss")
        difference = pivot["M2_temporal_defense"] - pivot["M1_static_defense"]
        game_result = difference.rename("delta_log_loss").reset_index()
        game_result["draw"] = draw
        game_rows.append(game_result)
        draw_rows.append({
            "draw": draw,
            "mean_difference": float(difference.mean()),
            "games_favoring_temporal": int((difference < 0).sum()),
        })
        if (draw + 1) % 10 == 0 or draw == 0:
            print(f"Monte Carlo draw {draw + 1}/{args.draws}", flush=True)

    draws = pd.DataFrame(draw_rows)
    pd.concat(game_rows, ignore_index=True).to_csv(args.output_dir / "game_differences.csv", index=False)
    draws.to_csv(args.output_dir / "draw_summary.csv", index=False)
    summary = {
        "draws": args.draws, "seed": args.seed, "shots_per_draw": len(data),
        "assumption": "Independent isotropic Gaussian x/y errors; predError is the 90% radial coverage bound, so axis SD = predError/sqrt(-2*ln(0.1)).",
        "mean_delta_log_loss": float(draws["mean_difference"].mean()),
        "mc_2.5pct": float(draws["mean_difference"].quantile(0.025)),
        "mc_97.5pct": float(draws["mean_difference"].quantile(0.975)),
        "fraction_temporal_better": float((draws["mean_difference"] < 0).mean()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.report_summary.parent.mkdir(parents=True, exist_ok=True)
    args.report_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
