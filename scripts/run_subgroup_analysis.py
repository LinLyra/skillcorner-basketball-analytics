from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skillcorner_basketball.io import load_dynamic_events


def cluster_bootstrap(values: pd.DataFrame, repetitions: int, seed: int) -> tuple[float, float, float]:
    per_game = values.groupby("game_id")["loss_difference"].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(per_game, size=(repetitions, len(per_game)), replace=True).mean(axis=1)
    return float(per_game.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Basketball-context subgroup analysis")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--robustness-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/subgroups"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/subgroups.png"))
    parser.add_argument("--repetitions", type=int, default=20_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    predictions = pd.read_csv(args.robustness_dir / "predictions.csv")
    predictions = predictions[
        (predictions["defender_definition"] == "fixed_release")
        & (predictions["quality_cohort"] == "all_eligible")
        & predictions["model"].isin(["M1_static_defense", "M2_temporal_defense"])
    ]
    wide = predictions.pivot(
        index=["game_id", "shot_id", "outcome"], columns="model", values="probability"
    ).reset_index()
    context_columns = [
        "game_id", "shot_id", "catch_and_shoot", "three", "blocked", "shot_type",
        "complex_shot_type", "recent_closing_speed_mps", "separation_gain_m",
    ]
    analysis = wide.merge(features[context_columns], on=["game_id", "shot_id"], how="left")

    closeout_shots: set[tuple[int, str]] = set()
    for game_id in sorted(analysis["game_id"].unique()):
        events = load_dynamic_events(
            args.data_root / str(game_id) / f"{game_id}_dynamic_events.json"
        )
        shots = events["shots"][["id", "touchId"]]
        closeout_touch_ids = set(events["closeouts"]["touchId"].dropna().astype(str))
        for shot in shots.to_dict("records"):
            if str(shot.get("touchId")) in closeout_touch_ids:
                closeout_shots.add((int(game_id), shot["id"]))
    analysis["official_closeout_context"] = [
        (int(game), shot) in closeout_shots
        for game, shot in zip(analysis["game_id"], analysis["shot_id"])
    ]
    analysis["pull_up"] = analysis["complex_shot_type"].fillna("").str.lower().str.contains("pullup")
    analysis["strong_close"] = analysis["recent_closing_speed_mps"] >= 1.0

    y = analysis["outcome"].to_numpy(dtype=float)
    static = np.clip(analysis["M1_static_defense"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    temporal = np.clip(analysis["M2_temporal_defense"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    analysis["static_loss"] = -(y * np.log(static) + (1 - y) * np.log(1 - static))
    analysis["temporal_loss"] = -(y * np.log(temporal) + (1 - y) * np.log(1 - temporal))
    analysis["loss_difference"] = analysis["temporal_loss"] - analysis["static_loss"]

    groups = {
        "All eligible shots": pd.Series(True, index=analysis.index),
        "Catch-and-shoot": analysis["catch_and_shoot"].eq(True),
        "Not catch-and-shoot": analysis["catch_and_shoot"].eq(False),
        "Pull-up": analysis["pull_up"],
        "Three-point": analysis["three"].eq(True),
        "Two-point": analysis["three"].eq(False),
        "Official closeout context": analysis["official_closeout_context"],
        "No official closeout": ~analysis["official_closeout_context"],
        "Closing speed ≥ 1 m/s": analysis["strong_close"],
        "Closing speed < 1 m/s": ~analysis["strong_close"],
    }
    rows = []
    for index, (name, mask) in enumerate(groups.items()):
        subset = analysis[mask].copy()
        mean, low, high = cluster_bootstrap(subset, args.repetitions, 20261024 + index)
        rows.append(
            {
                "subgroup": name,
                "shots": len(subset),
                "games": subset["game_id"].nunique(),
                "made_rate": subset["outcome"].mean(),
                "mean_delta_log_loss": mean,
                "ci95_low": low,
                "ci95_high": high,
                "games_favoring_temporal": int(
                    (subset.groupby("game_id")["loss_difference"].mean() < 0).sum()
                ),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "subgroup_results.csv", index=False)
    analysis.to_csv(args.output_dir / "shot_level_context.csv", index=False)

    plot = results.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6.2), constrained_layout=True)
    y_position = np.arange(len(plot))
    ax.errorbar(
        plot["mean_delta_log_loss"], y_position,
        xerr=np.vstack([
            plot["mean_delta_log_loss"] - plot["ci95_low"],
            plot["ci95_high"] - plot["mean_delta_log_loss"],
        ]),
        fmt="o", color="#255F85", ecolor="#7098B4", linewidth=2, capsize=3,
    )
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y_position, [f"{name}  (n={n})" for name, n in zip(plot["subgroup"], plot["shots"])])
    ax.set_xlabel("Temporal − static log loss (95% game-cluster bootstrap CI)")
    ax.set_title("Where does one-second temporal context help?")
    ax.grid(axis="x", alpha=0.2)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
