from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skillcorner_basketball.io import load_tracking


def player_key(value) -> str:
    return str(int(float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a static/temporal disagreement shot")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/study"))
    parser.add_argument("--robustness-dir", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("reports/media/top_disagreement.gif"))
    args = parser.parse_args()

    cases = pd.read_csv(args.robustness_dir / "top_model_disagreements.csv")
    case = cases.iloc[args.rank - 1]
    features = pd.read_csv(args.study_dir / "shot_features_1s.csv")
    feature = features[
        (features["game_id"] == case["game_id"]) & (features["shot_id"] == case["shot_id"])
    ].iloc[0]
    game_id = str(int(case["game_id"]))
    release = int(feature["release_frame"])
    _, entities = load_tracking(
        args.data_root / game_id / f"{game_id}_tracking_data.jsonl.gz", game_id
    )
    window = entities[entities["frame_idx"].between(release - 25, release)].copy()
    window["player_key"] = window["player_id"].map(lambda value: player_key(value) if pd.notna(value) else "ball")
    shooter_id = player_key(feature["shooter_id"])
    defender_id = player_key(feature["defender_id"])
    shooter = window[window["player_key"] == shooter_id].set_index("frame_idx")
    defender = window[window["player_key"] == defender_id].set_index("frame_idx")
    shared_frames = sorted(set(shooter.index) & set(defender.index))
    separation = np.array([
        np.hypot(
            defender.loc[frame, "x_m"] - shooter.loc[frame, "x_m"],
            defender.loc[frame, "y_m"] - shooter.loc[frame, "y_m"],
        )
        for frame in shared_frames
    ])
    seconds = (np.asarray(shared_frames) - release) / 25

    fig, (court, curve) = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    court.add_patch(plt.Rectangle((-14.3256, -7.62), 28.6512, 15.24, fill=False, color="#64748B"))
    court.axvline(0, color="#CBD5E1", linewidth=1)
    court.scatter([-12.4968, 12.4968], [0, 0], s=25, color="#F59E0B")
    court.set(xlim=(-14.7, 14.7), ylim=(-8, 8), aspect="equal")
    court.set_xticks([]); court.set_yticks([])
    court.set_title(f"{case['shot_id']} · {'made' if case['outcome'] else 'missed'}")

    curve.plot(seconds, separation, color="#CBD5E1", linewidth=2)
    curve.axvline(0, color="#111827", linewidth=1)
    curve.set_xlim(-1.02, 0.02)
    margin = max(0.15, float(np.ptp(separation)) * 0.15)
    curve.set_ylim(float(separation.min() - margin), float(separation.max() + margin))
    curve.set_xlabel("Seconds to release")
    curve.set_ylabel("Shooter–defender distance (m)")
    curve.grid(alpha=0.2)
    curve.set_title(
        f"Static make p={case['M1_static_defense']:.2f} · temporal p={case['M2_temporal_defense']:.2f}"
    )

    team_scatter = court.scatter([], [], s=38, color="#94A3B8", alpha=0.65)
    ball_scatter = court.scatter([], [], s=28, color="#F59E0B", zorder=4)
    shooter_scatter = court.scatter([], [], s=85, color="#2563EB", zorder=5, label="Shooter")
    defender_scatter = court.scatter([], [], s=85, color="#DC2626", zorder=5, label="Release defender")
    shooter_trail, = court.plot([], [], color="#2563EB", linewidth=3)
    defender_trail, = court.plot([], [], color="#DC2626", linewidth=3)
    distance_line, = curve.plot([], [], color="#7C3AED", linewidth=3)
    distance_marker = curve.scatter([], [], s=55, color="#7C3AED", zorder=5)
    clock = court.text(0.02, 0.97, "", transform=court.transAxes, va="top")
    court.legend(frameon=False, loc="upper center", ncol=2)

    def update(frame_number: int):
        frame = shared_frames[frame_number]
        current = window[window["frame_idx"] == frame]
        ordinary = current[(current["entity_type"] == "player") & ~current["player_key"].isin([shooter_id, defender_id])]
        ball = current[current["entity_type"] == "ball"]
        team_scatter.set_offsets(ordinary[["x_m", "y_m"]].to_numpy())
        ball_scatter.set_offsets(ball[["x_m", "y_m"]].to_numpy() if not ball.empty else np.empty((0, 2)))
        shooter_scatter.set_offsets(shooter.loc[[frame], ["x_m", "y_m"]].to_numpy())
        defender_scatter.set_offsets(defender.loc[[frame], ["x_m", "y_m"]].to_numpy())
        history = shared_frames[max(0, frame_number - 7):frame_number + 1]
        shooter_trail.set_data(shooter.loc[history, "x_m"], shooter.loc[history, "y_m"])
        defender_trail.set_data(defender.loc[history, "x_m"], defender.loc[history, "y_m"])
        distance_line.set_data(seconds[: frame_number + 1], separation[: frame_number + 1])
        distance_marker.set_offsets([[seconds[frame_number], separation[frame_number]]])
        clock.set_text(f"t = {seconds[frame_number]:.2f} s")
        return (
            team_scatter, ball_scatter, shooter_scatter, defender_scatter,
            shooter_trail, defender_trail, distance_line, distance_marker, clock,
        )

    movie = animation.FuncAnimation(fig, update, frames=len(shared_frames), interval=100, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(args.output, writer=animation.PillowWriter(fps=10), dpi=130)
    print(args.output)


if __name__ == "__main__":
    main()
