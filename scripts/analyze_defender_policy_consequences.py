from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skillcorner_basketball.io import load_tracking


FEATURES = [
    "release_distance_m", "recent_closing_speed_mps", "separation_gain_m",
    "pressure_duration_s", "mean_distance_m", "min_distance_m", "max_distance_m",
]


def normalise_id(value) -> str:
    return str(int(float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify consequences of defender policy")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--robustness-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/defender_policy"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/defender_policy_cases.png"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    nearest = pd.read_csv(args.robustness_dir / "shot_features_frame_nearest_1s.csv")
    hysteresis = pd.read_csv(
        args.robustness_dir / "shot_features_frame_nearest_hysteresis_0_3m_1s.csv"
    )
    base_cols = ["game_id", "shot_id", "outcome", "release_frame", "shooter_id", "defender_id"]
    common = fixed[base_cols + FEATURES].merge(
        nearest[["game_id", "shot_id", "unique_defenders_in_window"] + FEATURES],
        on=["game_id", "shot_id"], suffixes=("_fixed", "_nearest"), how="inner",
    )
    common = common.dropna(subset=[f"{name}_fixed" for name in FEATURES] + [f"{name}_nearest" for name in FEATURES])

    feature_rows = []
    for name in FEATURES:
        a, b = common[f"{name}_fixed"], common[f"{name}_nearest"]
        difference = b - a
        feature_rows.append({
            "feature": name, "n": len(common), "pearson": float(a.corr(b)),
            "spearman": float(a.corr(b, method="spearman")),
            "mean_absolute_difference": float(difference.abs().mean()),
            "p95_absolute_difference": float(difference.abs().quantile(.95)),
        })
    feature_summary = pd.DataFrame(feature_rows)
    feature_summary.to_csv(args.output_dir / "feature_consequences.csv", index=False)

    predictions = pd.read_csv(args.robustness_dir / "predictions.csv")
    predictions = predictions[
        (predictions["quality_cohort"] == "all_eligible")
        & (predictions["model"] == "M2_temporal_defense")
    ]
    wide = predictions.pivot_table(
        index=["game_id", "shot_id", "outcome"], columns="defender_definition",
        values="probability",
    ).dropna().reset_index()
    wide["probability_difference"] = wide["frame_nearest"] - wide["fixed_release"]
    wide["absolute_probability_difference"] = wide["probability_difference"].abs()
    wide = wide.merge(common, on=["game_id", "shot_id", "outcome"], how="left")
    wide["defender_switched"] = wide["unique_defenders_in_window"] > 1
    wide.to_csv(args.output_dir / "shot_consequences.csv", index=False)

    hysteresis_valid = hysteresis[hysteresis["status"] == "ok"]
    common_policy_scores = pd.read_csv(
        args.robustness_dir / "policy_comparison_common_scores.csv"
    )
    mean_policy_scores = common_policy_scores.groupby("model")[["log_loss", "brier", "auc"]].mean()
    prediction_summary = {
        "common_predictions": int(len(wide)),
        "pearson_probability": float(wide["fixed_release"].corr(wide["frame_nearest"])),
        "mean_absolute_probability_difference": float(wide["absolute_probability_difference"].mean()),
        "p95_absolute_probability_difference": float(wide["absolute_probability_difference"].quantile(.95)),
        "count_absolute_difference_over_0.05": int((wide["absolute_probability_difference"] > .05).sum()),
        "switch_rate": float(wide["defender_switched"].mean()),
        "mean_abs_difference_switched": float(wide.loc[wide["defender_switched"], "absolute_probability_difference"].mean()),
        "mean_abs_difference_not_switched": float(wide.loc[~wide["defender_switched"], "absolute_probability_difference"].mean()),
        "nearest_switch_rate_valid_trajectories": float(
            (nearest.loc[nearest["status"] == "ok", "unique_defenders_in_window"] > 1).mean()
        ),
        "hysteresis_0_3m_switch_rate_valid_trajectories": float(
            (hysteresis_valid["unique_defenders_in_window"] > 1).mean()
        ),
        "common_cohort_mean_scores": mean_policy_scores.to_dict(orient="index"),
        "interpretation": (
            "Descriptive sensitivity only: switching is not randomized and does not cause the "
            "prediction differences. On the identical held-out cohort, all policy log-loss "
            "differences are near zero and their paired game intervals cross zero."
        ),
    }
    (args.output_dir / "prediction_summary.json").write_text(json.dumps(prediction_summary, indent=2) + "\n")

    cases = wide.sort_values("absolute_probability_difference", ascending=False).head(3).copy()
    cases.to_csv(args.output_dir / "top_cases.csv", index=False)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5), constrained_layout=True)
    cache = {}
    for row_index, case in cases.reset_index(drop=True).iterrows():
        game_id = str(int(case["game_id"]))
        if game_id not in cache:
            _, cache[game_id] = load_tracking(args.data_root / game_id / f"{game_id}_tracking_data.jsonl.gz", game_id)
        entities = cache[game_id]
        release = int(case["release_frame"])
        window = entities[(entities["entity_type"] == "player") & entities["frame_idx"].between(release - 25, release)].copy()
        window["player_key"] = window["player_id"].map(normalise_id)
        shooter = window[window["player_key"] == normalise_id(case["shooter_id"])][["frame_idx", "x_m", "y_m", "team_side"]]
        release_defender = window[window["player_key"] == normalise_id(case["defender_id"])][["frame_idx", "x_m", "y_m"]]
        fixed_traj = shooter.merge(release_defender, on="frame_idx", suffixes=("_s", "_d"))
        fixed_traj["distance"] = np.hypot(fixed_traj["x_m_d"] - fixed_traj["x_m_s"], fixed_traj["y_m_d"] - fixed_traj["y_m_s"])
        opponent_side = "away" if shooter["team_side"].dropna().iloc[0] == "home" else "home"
        candidates = window[window["team_side"] == opponent_side][["frame_idx", "player_key", "x_m", "y_m"]]
        near = shooter.merge(candidates, on="frame_idx", suffixes=("_s", "_d"))
        near["distance"] = np.hypot(near["x_m_d"] - near["x_m_s"], near["y_m_d"] - near["y_m_s"])
        near = near.loc[near.groupby("frame_idx")["distance"].idxmin()].sort_values("frame_idx")

        court = axes[row_index, 0]
        court.add_patch(plt.Rectangle((-14.3256, -7.62), 28.6512, 15.24, fill=False, color="#64748B"))
        court.axvline(0, color="#CBD5E1", linewidth=1)
        court.scatter(window.loc[window["frame_idx"] == release, "x_m"], window.loc[window["frame_idx"] == release, "y_m"], color="#94A3B8", s=18)
        court.plot(shooter["x_m"], shooter["y_m"], color="#2563EB", linewidth=3, label="Shooter")
        court.plot(release_defender["x_m"], release_defender["y_m"], color="#DC2626", linewidth=3, label="Release defender")
        court.scatter(near["x_m_d"], near["y_m_d"], color="#F59E0B", s=16, label="Frame-nearest")
        court.set(xlim=(-14.7, 14.7), ylim=(-8, 8), aspect="equal")
        court.set_xticks([]); court.set_yticks([])
        court.set_title(f"Game {game_id} · shot {case['shot_id']}")
        if row_index == 0: court.legend(frameon=False, fontsize=8, ncol=3)

        curve = axes[row_index, 1]
        curve.plot((fixed_traj["frame_idx"] - release) / 25, fixed_traj["distance"], color="#DC2626", linewidth=2.5, label="Fixed release defender")
        curve.plot((near["frame_idx"] - release) / 25, near["distance"], color="#F59E0B", linewidth=2.5, label="Frame-nearest")
        curve.set(xlabel="Seconds to release", ylabel="Distance (m)")
        curve.grid(alpha=.2)
        curve.set_title(f"make p {case['fixed_release']:.2f} → {case['frame_nearest']:.2f} (Δ {case['probability_difference']:+.2f})")
        if row_index == 0: curve.legend(frameon=False, fontsize=8)
    fig.suptitle("When the defender definition changes the analytical story", fontsize=15)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    print(json.dumps(prediction_summary, indent=2))


if __name__ == "__main__":
    main()
