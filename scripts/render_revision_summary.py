from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def main() -> None:
    features = pd.read_csv("outputs/study/shot_features_1s.csv")
    data = features[(features["status"] == "ok") & (~features["fouled"]) & features["release_distance_aligned_5cm"].eq(True)]
    base_rows = []
    for game in sorted(data["game_id"].unique()):
        train, test = data[data["game_id"] != game], data[data["game_id"] == game]
        probability = np.repeat(train["outcome"].mean(), len(test))
        base_rows.append({"game_id": game, "log_loss": log_loss(test["outcome"], probability, labels=[0, 1])})
    base = pd.DataFrame(base_rows)
    study = pd.read_csv("outputs/study/logo_scores.csv")
    study = study[study["window_seconds"] == 1]
    official = pd.read_csv("outputs/official_benchmark/scores.csv")
    nested = pd.read_csv("outputs/nested_models/scores.csv")
    means = {
        "Training-game\nbase rate": base["log_loss"].mean(),
        "M0 location +\ncontext": study.loc[study["model"] == "M0_location_context", "log_loss"].mean(),
        "M1 static\ndefense": study.loc[study["model"] == "M1_static_defense", "log_loss"].mean(),
        "M2 temporal\ndefense": study.loc[study["model"] == "M2_temporal_defense", "log_loss"].mean(),
        "Monotonic tree\nstatic": nested.loc[nested["model"] == "MonotonicTree_static", "log_loss"].mean(),
        "Official SQ\n(reference*)": official.loc[official["model"] == "Official_Shot_Quality_direct", "log_loss"].mean(),
    }
    Path("outputs/effect_context").mkdir(parents=True, exist_ok=True)
    base.to_csv("outputs/effect_context/base_rate_scores.csv", index=False)
    summary = {"mean_log_loss": means, "improvement_from_base_rate": {k: means["Training-game\nbase rate"] - v for k, v in means.items()}}
    Path("outputs/effect_context/summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    inference = json.loads(Path("outputs/inference_checks/summary.json").read_text())
    policy = json.loads(Path("outputs/defender_policy/prediction_summary.json").read_text())
    mc = json.loads(Path("outputs/pred_error_mc/summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    colors = ["#94A3B8", "#457B9D", "#2A9D8F", "#7C3AED", "#264653", "#F59E0B"]
    axes[0].barh(list(means.keys()), list(means.values()), color=colors)
    axes[0].set_xlabel("Leave-one-game-out log loss")
    axes[0].set_xlim(.59, .70)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Predictive scale, not just rank")
    axes[0].text(.01, -.28, "*External production reference; training overlap is unknown.", transform=axes[0].transAxes, fontsize=8)

    game_diff = pd.read_csv("outputs/inference_checks/game_differences.csv").sort_values("delta_log_loss")
    axes[1].barh(game_diff["game_id"].astype(str), game_diff["delta_log_loss"], color=np.where(game_diff["delta_log_loss"] < 0, "#2A9D8F", "#D1495B"))
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_xlabel("M2 − M1 log loss")
    axes[1].set_title(f"10 games · exact sign-flip p={inference['exact_sign_flip_two_sided_p']:.3f}")

    labels = ["Observed\nfixed coords", "predError MC\nmean"]
    values = [inference["observed_delta_log_loss"], mc["mean_delta_log_loss"]]
    low = [0, values[1] - mc["mc_2.5pct"]]
    high = [0, mc["mc_97.5pct"] - values[1]]
    axes[2].errorbar(labels, values, yerr=[low, high], fmt="o", markersize=8, capsize=6, color="#7C3AED")
    axes[2].axhline(0, color="#111827", linewidth=1)
    axes[2].set_ylabel("M2 − M1 log loss")
    axes[2].set_title("Position-error sensitivity")
    axes[2].text(
        .02, .03,
        f"Nearest switches: {policy['nearest_switch_rate_valid_trajectories']:.1%}\n"
        f"0.3 m hysteresis: {policy['hysteresis_0_3m_switch_rate_valid_trajectories']:.1%}\n"
        "No detectable policy loss difference",
        transform=axes[2].transAxes, fontsize=9,
    )
    fig.suptitle("Beyond the Snapshot: what survives stricter scrutiny", fontsize=16)
    out = Path("reports/figures/revision_summary.png")
    fig.savefig(out, dpi=230, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
