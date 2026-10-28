from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise_id(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _slope_per_second(frame_idx: np.ndarray, values: np.ndarray, fps: float) -> float:
    """Return an OLS slope in value units/second."""
    if len(values) < 2 or np.ptp(frame_idx) == 0:
        return np.nan
    seconds = (frame_idx - frame_idx[-1]) / fps
    return float(np.polyfit(seconds, values, deg=1)[0])


def build_shot_temporal_features(
    shots: pd.DataFrame,
    entities: pd.DataFrame,
    *,
    game_id: int | str,
    window_seconds: float = 2.0,
    recent_seconds: float = 0.4,
    fps: float = 25.0,
    pressure_distance_m: float = 1.5,
    defender_mode: str = "fixed_release",
    defender_switch_margin_m: float = 0.3,
    smoothing_frames: int = 1,
    coordinate_noise_seed: int | None = None,
    pred_error_coverage: float = 0.90,
) -> pd.DataFrame:
    """Build one analysis row per shot from a defender-distance trajectory.

    ``fixed_release`` holds ``closestDefId`` fixed through the window.
    ``frame_nearest`` selects the nearest opponent independently in each frame.
    ``frame_nearest_hysteresis`` starts from the official release defender and,
    working backward, changes identity only when another defender is at least
    ``defender_switch_margin_m`` closer. This suppresses near-tie switches.
    """
    required = {
        "id",
        "startFrame",
        "shooterId",
        "closestDefId",
        "closestDefDist",
        "outcome",
    }
    missing = sorted(required - set(shots.columns))
    if missing:
        raise ValueError(f"shots table missing required columns: {missing}")
    if window_seconds <= 0 or recent_seconds <= 0 or fps <= 0:
        raise ValueError("window_seconds, recent_seconds, and fps must be positive")
    if defender_mode not in {"fixed_release", "frame_nearest", "frame_nearest_hysteresis"}:
        raise ValueError(
            "defender_mode must be 'fixed_release', 'frame_nearest', or "
            "'frame_nearest_hysteresis'"
        )
    if defender_switch_margin_m < 0:
        raise ValueError("defender_switch_margin_m must be non-negative")
    if smoothing_frames < 1 or smoothing_frames % 2 == 0:
        raise ValueError("smoothing_frames must be a positive odd integer")
    if not 0 < pred_error_coverage < 1:
        raise ValueError("pred_error_coverage must be between zero and one")
    rng = np.random.default_rng(coordinate_noise_seed) if coordinate_noise_seed is not None else None
    # SkillCorner describes predError as a radial coverage bound.  Under the
    # explicit isotropic-Gaussian sensitivity model below, radius R satisfies
    # P(||error|| <= R) = coverage for a Rayleigh-distributed radius.
    radius_to_axis_sd = np.sqrt(-2.0 * np.log(1.0 - pred_error_coverage))

    players = entities[entities["entity_type"] == "player"].copy()
    players["player_key"] = players["player_id"].map(_normalise_id)
    players = players.sort_values(["player_key", "frame_idx"])
    by_player = {
        key: group.set_index("frame_idx", drop=False)
        for key, group in players.groupby("player_key", sort=False)
    }
    window_frames = int(round(window_seconds * fps))
    recent_frames = int(round(recent_seconds * fps))
    expected_frames = window_frames + 1
    rows: list[dict] = []

    for shot in shots.to_dict("records"):
        release_frame = int(shot["startFrame"])
        start_frame = release_frame - window_frames
        shooter_key = _normalise_id(shot.get("shooterId"))
        defender_key = _normalise_id(shot.get("closestDefId"))
        shooter = by_player.get(shooter_key)
        defender = by_player.get(defender_key)

        status = "ok"
        trajectory = pd.DataFrame()
        if shooter is None:
            status = "missing_shooter"
        elif defender_mode in {"fixed_release", "frame_nearest_hysteresis"} and defender is None:
            status = "missing_defender"
        else:
            shooter_window = shooter.loc[
                shooter.index.to_series().between(start_frame, release_frame),
                ["frame_idx", "x_m", "y_m", "is_detected", "pred_error_m"],
            ].reset_index(drop=True).rename(
                columns={
                    "x_m": "shooter_x_m",
                    "y_m": "shooter_y_m",
                    "is_detected": "shooter_detected",
                    "pred_error_m": "shooter_pred_error_m",
                }
            )
            if defender_mode == "fixed_release":
                defender_window = defender.loc[
                    defender.index.to_series().between(start_frame, release_frame),
                    ["frame_idx", "player_key", "x_m", "y_m", "is_detected", "pred_error_m"],
                ].reset_index(drop=True).rename(
                    columns={
                        "player_key": "frame_defender_id",
                        "x_m": "defender_x_m",
                        "y_m": "defender_y_m",
                        "is_detected": "defender_detected",
                        "pred_error_m": "defender_pred_error_m",
                    }
                )
                trajectory = shooter_window.merge(
                    defender_window, on="frame_idx", how="inner"
                )
            else:
                shooter_side = shooter["team_side"].dropna()
                if shooter_side.empty:
                    status = "unknown_shooter_team"
                    trajectory = pd.DataFrame()
                else:
                    opponent_side = "away" if shooter_side.iloc[0] == "home" else "home"
                    candidates = players[
                        (players["team_side"] == opponent_side)
                        & players["frame_idx"].between(start_frame, release_frame)
                    ][
                        ["frame_idx", "player_key", "x_m", "y_m", "is_detected", "pred_error_m"]
                    ].rename(
                        columns={
                            "player_key": "frame_defender_id",
                            "x_m": "defender_x_m",
                            "y_m": "defender_y_m",
                            "is_detected": "defender_detected",
                            "pred_error_m": "defender_pred_error_m",
                        }
                    )
                    candidate_pairs = shooter_window.merge(
                        candidates, on="frame_idx", how="inner"
                    ).dropna(
                        subset=["shooter_x_m", "shooter_y_m", "defender_x_m", "defender_y_m"]
                    )
                    candidate_pairs["candidate_distance_m"] = np.hypot(
                        candidate_pairs["defender_x_m"] - candidate_pairs["shooter_x_m"],
                        candidate_pairs["defender_y_m"] - candidate_pairs["shooter_y_m"],
                    )
                    nearest_index = candidate_pairs.groupby("frame_idx")["candidate_distance_m"].idxmin()
                    if defender_mode == "frame_nearest":
                        trajectory = candidate_pairs.loc[nearest_index].copy()
                    else:
                        # Anchor on the official release defender, then traverse
                        # backward so near ties do not create identity flicker.
                        selected_indices: list[int] = []
                        current_id = defender_key
                        for frame_idx in sorted(candidate_pairs["frame_idx"].unique(), reverse=True):
                            frame_candidates = candidate_pairs[candidate_pairs["frame_idx"] == frame_idx]
                            nearest_row = frame_candidates.loc[frame_candidates["candidate_distance_m"].idxmin()]
                            current_rows = frame_candidates[frame_candidates["frame_defender_id"] == current_id]
                            if frame_idx == release_frame and not current_rows.empty:
                                chosen = current_rows.iloc[0]
                            elif current_rows.empty:
                                chosen = nearest_row
                            else:
                                current_row = current_rows.iloc[0]
                                improvement = current_row["candidate_distance_m"] - nearest_row["candidate_distance_m"]
                                chosen = nearest_row if improvement >= defender_switch_margin_m else current_row
                            current_id = chosen["frame_defender_id"]
                            selected_indices.append(int(chosen.name))
                        trajectory = candidate_pairs.loc[selected_indices].copy()
            trajectory = trajectory.dropna(
                subset=["shooter_x_m", "shooter_y_m", "defender_x_m", "defender_y_m"]
            ).sort_values("frame_idx")
            if trajectory.empty:
                status = "no_overlap"
            elif release_frame not in set(trajectory["frame_idx"]):
                status = "missing_release_frame"

        base = {
            "game_id": game_id,
            "shot_id": shot["id"],
            "release_frame": release_frame,
            "shooter_id": shooter_key,
            "defender_id": defender_key,
            "defender_mode": defender_mode,
            "defender_switch_margin_m": defender_switch_margin_m,
            "smoothing_frames": smoothing_frames,
            "outcome": bool(shot["outcome"]),
            "fouled": bool(shot.get("fouled", False)),
            "blocked": bool(shot.get("blocked", False)),
            "three": bool(shot.get("three", False)),
            "shot_type": shot.get("shotType"),
            "complex_shot_type": shot.get("complexShotType"),
            "region": shot.get("region"),
            "catch_and_shoot": bool(shot.get("catchAndShoot", False)),
            "dribbles_before": shot.get("dribblesBefore"),
            "shot_distance_ft": shot.get("distance"),
            "shot_clock_s": shot.get("shotClock"),
            "official_shot_quality": shot.get("shotQuality"),
            "official_release_distance_m": float(shot["closestDefDist"]) * 0.3048
            if pd.notna(shot.get("closestDefDist"))
            else np.nan,
            "window_seconds": window_seconds,
            "status": status,
            "expected_frame_count": expected_frames,
            "observed_frame_count": int(len(trajectory)),
            "frame_coverage": float(len(trajectory) / expected_frames),
            "unique_defenders_in_window": int(trajectory["frame_defender_id"].nunique())
            if "frame_defender_id" in trajectory
            else 0,
        }
        if trajectory.empty:
            rows.append(base)
            continue

        if rng is not None:
            for prefix in ("shooter", "defender"):
                error = trajectory[f"{prefix}_pred_error_m"].fillna(0).clip(lower=0)
                axis_sd = error.to_numpy(dtype=float) / radius_to_axis_sd
                trajectory[f"{prefix}_x_m"] += rng.normal(0.0, axis_sd)
                trajectory[f"{prefix}_y_m"] += rng.normal(0.0, axis_sd)

        raw_separation = np.asarray(np.hypot(
            trajectory["defender_x_m"] - trajectory["shooter_x_m"],
            trajectory["defender_y_m"] - trajectory["shooter_y_m"],
        ), dtype=float)
        raw_release_distance = float(
            raw_separation[trajectory["frame_idx"].to_numpy() == release_frame][-1]
        ) if release_frame in set(trajectory["frame_idx"]) else np.nan
        if smoothing_frames > 1:
            coordinate_columns = [
                "shooter_x_m", "shooter_y_m", "defender_x_m", "defender_y_m"
            ]
            trajectory[coordinate_columns] = trajectory[coordinate_columns].rolling(
                smoothing_frames, center=True, min_periods=1
            ).median()
        trajectory["separation_m"] = np.hypot(
            trajectory["defender_x_m"] - trajectory["shooter_x_m"],
            trajectory["defender_y_m"] - trajectory["shooter_y_m"],
        )
        trajectory["combined_pred_error_m"] = (
            trajectory["shooter_pred_error_m"].fillna(0)
            + trajectory["defender_pred_error_m"].fillna(0)
        )
        trajectory["both_detected"] = (
            trajectory["shooter_detected"].fillna(False).astype(bool)
            & trajectory["defender_detected"].fillna(False).astype(bool)
        )
        frames = trajectory["frame_idx"].to_numpy(dtype=float)
        separation = trajectory["separation_m"].to_numpy(dtype=float)
        recent = trajectory[trajectory["frame_idx"] >= release_frame - recent_frames]
        recent_sep = recent["separation_m"].to_numpy(dtype=float)
        recent_frame_idx = recent["frame_idx"].to_numpy(dtype=float)
        release_distance = raw_release_distance

        base.update(
            {
                "release_distance_m": release_distance,
                "window_start_distance_m": float(separation[0]),
                "mean_distance_m": float(np.mean(separation)),
                "min_distance_m": float(np.min(separation)),
                "max_distance_m": float(np.max(separation)),
                "separation_gain_m": float(release_distance - separation[0])
                if np.isfinite(release_distance)
                else np.nan,
                "net_closing_speed_mps": -_slope_per_second(frames, separation, fps),
                "recent_closing_speed_mps": -_slope_per_second(
                    recent_frame_idx, recent_sep, fps
                ),
                "pressure_duration_s": float(
                    np.sum(separation <= pressure_distance_m) / fps
                ),
                "both_detected_rate": float(trajectory["both_detected"].mean()),
                "mean_combined_pred_error_m": float(
                    trajectory["combined_pred_error_m"].mean()
                ),
                "max_combined_pred_error_m": float(
                    trajectory["combined_pred_error_m"].max()
                ),
            }
        )
        base["release_distance_error_m"] = (
            abs(base["release_distance_m"] - base["official_release_distance_m"])
            if np.isfinite(base["release_distance_m"])
            and np.isfinite(base["official_release_distance_m"])
            else np.nan
        )
        base["release_distance_aligned_5cm"] = bool(
            base["release_distance_error_m"] <= 0.05
        ) if np.isfinite(base["release_distance_error_m"]) else pd.NA
        rows.append(base)

    return pd.DataFrame(rows)
