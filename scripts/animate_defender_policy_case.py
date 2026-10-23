from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skillcorner_basketball.io import load_tracking


def key(value) -> str:
    return str(int(float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a defender-policy disagreement")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--cases", type=Path, default=Path("outputs/defender_policy/top_cases.csv"))
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    case = pd.read_csv(args.cases).iloc[args.rank - 1]
    game_id, release = str(int(case["game_id"])), int(case["release_frame"])
    _, entities = load_tracking(args.data_root / game_id / f"{game_id}_tracking_data.jsonl.gz", game_id)
    window = entities[(entities["entity_type"] == "player") & entities["frame_idx"].between(release - 25, release)].copy()
    window["key"] = window["player_id"].map(key)
    shooter_id, fixed_id = key(case["shooter_id"]), key(case["defender_id"])
    shooter = window[window["key"] == shooter_id].set_index("frame_idx")
    fixed = window[window["key"] == fixed_id].set_index("frame_idx")
    opponent = "away" if shooter["team_side"].dropna().iloc[0] == "home" else "home"
    frames = sorted(set(shooter.index) & set(fixed.index))
    nearest = {}
    fixed_distance, nearest_distance = [], []
    for frame in frames:
        opponents = window[(window["frame_idx"] == frame) & (window["team_side"] == opponent)].copy()
        sx, sy = shooter.loc[frame, ["x_m", "y_m"]]
        opponents["distance"] = np.hypot(opponents["x_m"] - sx, opponents["y_m"] - sy)
        chosen = opponents.loc[opponents["distance"].idxmin()]
        nearest[frame] = chosen
        fixed_distance.append(np.hypot(fixed.loc[frame, "x_m"] - sx, fixed.loc[frame, "y_m"] - sy))
        nearest_distance.append(chosen["distance"])
    seconds = (np.asarray(frames) - release) / 25

    fig, (court, curve) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    court.add_patch(plt.Rectangle((-14.3256, -7.62), 28.6512, 15.24, fill=False, color="#64748B"))
    court.axvline(0, color="#CBD5E1", linewidth=1)
    court.set(xlim=(-14.7, 14.7), ylim=(-8, 8), aspect="equal")
    court.set_xticks([]); court.set_yticks([])
    curve.plot(seconds, fixed_distance, color="#DC2626", alpha=.25)
    curve.plot(seconds, nearest_distance, color="#F59E0B", alpha=.25)
    curve.set(xlabel="Seconds to release", ylabel="Distance (m)")
    curve.grid(alpha=.2)
    all_players = court.scatter([], [], color="#94A3B8", s=25)
    shot = court.scatter([], [], color="#2563EB", s=90, label="Shooter")
    release_def = court.scatter([], [], color="#DC2626", s=90, label="Release defender")
    near_def = court.scatter([], [], facecolors="none", edgecolors="#F59E0B", linewidths=2.5, s=150, label="Frame-nearest")
    fixed_line, = curve.plot([], [], color="#DC2626", linewidth=3, label="Fixed")
    near_line, = curve.plot([], [], color="#F59E0B", linewidth=3, label="Nearest")
    court.legend(frameon=False, fontsize=8, ncol=3)
    curve.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Game {game_id} · {case['shot_id']} · make p {case['fixed_release']:.2f} → {case['frame_nearest']:.2f}")

    def update(i):
        frame = frames[i]
        current = window[window["frame_idx"] == frame]
        all_players.set_offsets(current[["x_m", "y_m"]].to_numpy())
        shot.set_offsets(shooter.loc[[frame], ["x_m", "y_m"]].to_numpy())
        release_def.set_offsets(fixed.loc[[frame], ["x_m", "y_m"]].to_numpy())
        chosen = nearest[frame]
        near_def.set_offsets([[chosen["x_m"], chosen["y_m"]]])
        fixed_line.set_data(seconds[:i + 1], fixed_distance[:i + 1])
        near_line.set_data(seconds[:i + 1], nearest_distance[:i + 1])
        court.set_title(f"t={seconds[i]:.2f}s · nearest #{chosen['key']}")
        return all_players, shot, release_def, near_def, fixed_line, near_line

    output = args.output or Path(f"reports/media/defender_policy_case_{args.rank}.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.FuncAnimation(fig, update, frames=len(frames), interval=100).save(
        output, writer=animation.PillowWriter(fps=10), dpi=115
    )
    print(output)


if __name__ == "__main__":
    main()
