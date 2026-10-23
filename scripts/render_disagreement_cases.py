from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skillcorner_basketball.io import load_tracking


def normalise_id(value) -> str:
    return str(int(float(value)))


def draw_court(ax) -> None:
    ax.add_patch(plt.Rectangle((-14.3256, -7.62), 28.6512, 15.24, fill=False, color="#64748B"))
    ax.axvline(0, color="#CBD5E1", linewidth=1)
    ax.scatter([-12.4968, 12.4968], [0, 0], s=20, color="#F59E0B", zorder=3)
    ax.set_xlim(-14.7, 14.7)
    ax.set_ylim(-8.0, 8.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--robustness-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--output", type=Path, default=Path("reports/figures/disagreement_cases.png"))
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args()
    cases = pd.read_csv(args.robustness_dir / "top_model_disagreements.csv").head(args.cases)
    features = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    enriched = cases.merge(features, on=["game_id", "shot_id", "outcome"], how="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table_output = Path("reports/tables/top_model_disagreements.csv")
    table_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(table_output, index=False)

    fig, axes = plt.subplots(len(enriched), 2, figsize=(12, 3.8 * len(enriched)), constrained_layout=True)
    if len(enriched) == 1:
        axes = np.array([axes])
    cache = {}
    for row_index, case in enriched.reset_index(drop=True).iterrows():
        game_id = str(int(case["game_id"]))
        if game_id not in cache:
            _, cache[game_id] = load_tracking(
                args.data_root / game_id / f"{game_id}_tracking_data.jsonl.gz", game_id
            )
        entities = cache[game_id]
        release = int(case["release_frame"])
        window = entities[
            (entities["entity_type"] == "player")
            & entities["frame_idx"].between(release - 25, release)
        ].copy()
        window["player_key"] = window["player_id"].map(normalise_id)
        shooter = window[window["player_key"] == normalise_id(case["shooter_id"])]
        defender = window[window["player_key"] == normalise_id(case["defender_id"])]

        ax = axes[row_index, 0]
        draw_court(ax)
        release_players = window[window["frame_idx"] == release]
        ax.scatter(release_players["x_m"], release_players["y_m"], color="#94A3B8", s=18, alpha=0.55)
        ax.plot(shooter["x_m"], shooter["y_m"], color="#2563EB", linewidth=3, label="Shooter")
        ax.plot(defender["x_m"], defender["y_m"], color="#DC2626", linewidth=3, label="Release defender")
        ax.scatter(shooter["x_m"].iloc[-1], shooter["y_m"].iloc[-1], color="#2563EB", s=50)
        ax.scatter(defender["x_m"].iloc[-1], defender["y_m"].iloc[-1], color="#DC2626", s=50)
        ax.set_title(f"{case['shot_id']} · {'made' if case['outcome'] else 'missed'}")
        if row_index == 0:
            ax.legend(frameon=False, loc="upper center", ncol=2)

        trajectory = shooter[["frame_idx", "x_m", "y_m"]].merge(
            defender[["frame_idx", "x_m", "y_m"]], on="frame_idx", suffixes=("_shooter", "_defender")
        )
        trajectory["distance_m"] = np.hypot(
            trajectory["x_m_defender"] - trajectory["x_m_shooter"],
            trajectory["y_m_defender"] - trajectory["y_m_shooter"],
        )
        trajectory["seconds_to_release"] = (trajectory["frame_idx"] - release) / 25
        ax = axes[row_index, 1]
        ax.plot(trajectory["seconds_to_release"], trajectory["distance_m"], color="#7C3AED", linewidth=3)
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_xlabel("Seconds to release")
        ax.set_ylabel("Shooter–defender distance (m)")
        ax.grid(alpha=0.2)
        ax.set_title(
            f"Static p={case['M1_static_defense']:.2f} · temporal p={case['M2_temporal_defense']:.2f}"
        )

    fig.suptitle("Largest out-of-game static–temporal model disagreements", fontsize=15)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
