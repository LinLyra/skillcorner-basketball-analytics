from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def require(path: str) -> Path:
    target = ROOT / path
    if not target.exists() or target.stat().st_size == 0:
        raise AssertionError(f"Missing release artifact: {path}")
    return target


def main() -> None:
    required = [
        "reports/final_report.md",
        "reports/analytics_cup_submission.md",
        "reports/figures/final_summary.png",
        "reports/figures/negative_control.png",
        "reports/figures/subgroups.png",
        "reports/figures/inference_checks.png",
        "reports/figures/defender_policy_cases.png",
        "reports/figures/revision_summary.png",
        "reports/media/top_disagreement.gif",
        "reports/media/defender_policy_case_1.gif",
        "reports/media/defender_policy_case_2.gif",
        "reports/media/defender_policy_case_3.gif",
        "outputs/study/shot_features_1s.csv",
        "outputs/robustness/paired_bootstrap.csv",
        "outputs/official_benchmark/scores.csv",
        "outputs/inference_checks/summary.json",
        "outputs/nested_models/paired_bootstrap.csv",
        "outputs/pred_error_mc/summary.json",
        "outputs/defender_policy/prediction_summary.json",
        "outputs/robustness/policy_comparison_common_bootstrap.csv",
    ]
    for path in required:
        require(path)

    lock = json.loads(require("datasets.lock.json").read_text())
    assert len(lock["games"]) == 10, "Expected ten locked games"
    features = pd.read_csv(require("outputs/study/shot_features_1s.csv"))
    assert len(features) == 1459, "Unexpected shot count"
    eligible = features[
        (features["status"] == "ok")
        & (~features["fouled"])
        & features["release_distance_aligned_5cm"].eq(True)
    ]
    assert len(eligible) == 1211, "Unexpected primary cohort size"
    official_scores = pd.read_csv(require("outputs/official_benchmark/scores.csv"))
    means = official_scores.groupby("model")["log_loss"].mean()
    assert means["Official_Shot_Quality_direct"] < means["M2_temporal_defense"]
    robustness = pd.read_csv(require("outputs/robustness/paired_bootstrap.csv"))
    primary = robustness[
        (robustness["metric"] == "log_loss")
        & (robustness["defender_definition"] == "fixed_release")
        & (robustness["quality_cohort"] == "all_eligible")
    ].iloc[0]
    assert primary["ci95_low"] < 0 < primary["ci95_high"]
    inference = json.loads(require("outputs/inference_checks/summary.json").read_text())
    assert inference["exact_sign_flip_two_sided_p"] > 0.05
    assert inference["conditional_fraction_as_good_or_better"] > 0.05
    assert inference["exact_sign_flip_ci95_low"] < -0.006
    assert inference["exact_sign_flip_ci95_high"] > 0
    policy = json.loads(require("outputs/defender_policy/prediction_summary.json").read_text())
    assert policy["mean_abs_difference_switched"] > policy["mean_abs_difference_not_switched"]
    assert (
        policy["hysteresis_0_3m_switch_rate_valid_trajectories"]
        < policy["nearest_switch_rate_valid_trajectories"]
    )
    policy_comparison = pd.read_csv(
        require("outputs/robustness/policy_comparison_common_bootstrap.csv")
    )
    policy_logloss = policy_comparison[policy_comparison["metric"] == "log_loss"]
    assert ((policy_logloss["ci95_low"] < 0) & (policy_logloss["ci95_high"] > 0)).all()
    mc = json.loads(require("outputs/pred_error_mc/summary.json").read_text())
    assert mc["draws"] >= 200
    readme = require("README.md").read_text()
    assert len(readme.split()) <= 1_000
    assert readme.count("![") <= 2
    print("Release validation passed: 10 games, 1,459 shots, 1,211-shot primary cohort.")


if __name__ == "__main__":
    main()
