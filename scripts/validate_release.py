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
        "reports/nba_portfolio_brief.md",
        "reports/figures/final_summary.png",
        "reports/figures/negative_control.png",
        "reports/figures/subgroups.png",
        "reports/media/top_disagreement.gif",
        "outputs/study/shot_features_1s.csv",
        "outputs/robustness/paired_bootstrap.csv",
        "outputs/official_benchmark/scores.csv",
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
    print("Release validation passed: 10 games, 1,459 shots, 1,211-shot primary cohort.")


if __name__ == "__main__":
    main()
