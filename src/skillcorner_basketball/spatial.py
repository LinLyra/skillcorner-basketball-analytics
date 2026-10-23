from __future__ import annotations

import numpy as np
import pandas as pd


HOOP_POSITION_M = np.array([-40.75 * 0.3048, 0.0])


def _normalise_id(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(int(float(value)))


def _frame_geometry(
    frame_players: pd.DataFrame, shooter_id: str, shooter_side: str
) -> dict[str, float]:
    if frame_players.empty or "player_key" not in frame_players:
        return {}
    shooter_rows = frame_players[frame_players["player_key"] == shooter_id]
    if shooter_rows.empty:
        return {}
    shooter = shooter_rows.iloc[0][["x_m", "y_m"]].to_numpy(dtype=float)
    defenders = frame_players[frame_players["team_side"] != shooter_side].dropna(
        subset=["x_m", "y_m"]
    )
    teammates = frame_players[
        (frame_players["team_side"] == shooter_side)
        & (frame_players["player_key"] != shooter_id)
    ].dropna(subset=["x_m", "y_m"])
    if defenders.empty:
        return {}
    defense_xy = defenders[["x_m", "y_m"]].to_numpy(dtype=float)
    defense_vectors = defense_xy - shooter
    distances = np.linalg.norm(defense_vectors, axis=1)
    order = np.argsort(distances)
    hoop_vector = HOOP_POSITION_M - shooter
    hoop_distance = np.linalg.norm(hoop_vector)
    unit_hoop = hoop_vector / hoop_distance if hoop_distance else np.array([0.0, 0.0])
    projections = defense_vectors @ unit_hoop
    perpendicular = np.linalg.norm(
        defense_vectors - np.outer(projections, unit_hoop), axis=1
    )
    valid_lane = (projections > 0) & (projections < hoop_distance)
    cosine = np.divide(
        defense_vectors @ unit_hoop,
        distances,
        out=np.zeros_like(distances),
        where=distances > 0,
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    nearest = order[0]
    teammate_distances = (
        np.linalg.norm(teammates[["x_m", "y_m"]].to_numpy(dtype=float) - shooter, axis=1)
        if not teammates.empty
        else np.array([])
    )
    return {
        "nearest_defender_distance_m": float(distances[nearest]),
        "second_defender_distance_m": float(distances[order[1]]) if len(order) > 1 else np.nan,
        "third_defender_distance_m": float(distances[order[2]]) if len(order) > 2 else np.nan,
        "defenders_within_1_5m": int(np.sum(distances <= 1.5)),
        "defenders_within_3m": int(np.sum(distances <= 3.0)),
        "nearest_defender_angle_to_hoop_deg": float(angles[nearest]),
        "defenders_in_30deg_cone": int(np.sum((angles <= 30) & valid_lane)),
        "min_lane_obstruction_m": float(np.min(perpendicular[valid_lane]))
        if np.any(valid_lane)
        else np.nan,
        "defensive_centroid_distance_m": float(np.linalg.norm(defense_xy.mean(axis=0) - shooter)),
        "nearest_teammate_distance_m": float(np.min(teammate_distances))
        if len(teammate_distances)
        else np.nan,
        "mean_teammate_distance_m": float(np.mean(teammate_distances))
        if len(teammate_distances)
        else np.nan,
        "release_defender_count": int(len(defenders)),
        "release_teammate_count": int(len(teammates)),
    }


def build_shot_spatial_features(
    shots: pd.DataFrame,
    entities: pd.DataFrame,
    *,
    game_id: int | str,
    history_seconds: float = 1.0,
    fps: float = 25.0,
) -> pd.DataFrame:
    """Construct release geometry and help-defense collapse features per shot."""
    players = entities[entities["entity_type"] == "player"].copy()
    players["player_key"] = players["player_id"].map(_normalise_id)
    by_frame = {int(frame): group for frame, group in players.groupby("frame_idx", sort=False)}
    rows = []
    history_frames = int(round(history_seconds * fps))
    for shot in shots.to_dict("records"):
        release_frame = int(shot["startFrame"])
        start_frame = release_frame - history_frames
        shooter_id = _normalise_id(shot.get("shooterId"))
        release_players = by_frame.get(release_frame, pd.DataFrame())
        start_players = by_frame.get(start_frame, pd.DataFrame())
        shooter_release = (
            release_players[release_players["player_key"] == shooter_id]
            if not release_players.empty
            else pd.DataFrame()
        )
        status = "ok"
        if shooter_release.empty:
            status = "missing_release_shooter"
            release_geometry = {}
            start_geometry = {}
        else:
            shooter_side = shooter_release.iloc[0]["team_side"]
            release_geometry = _frame_geometry(release_players, shooter_id, shooter_side)
            start_geometry = _frame_geometry(start_players, shooter_id, shooter_side)
            if not release_geometry:
                status = "missing_release_geometry"
            elif not start_geometry:
                status = "missing_start_geometry"
        row = {"game_id": game_id, "shot_id": shot["id"], "spatial_status": status}
        row.update({f"release_{key}": value for key, value in release_geometry.items()})
        row.update({f"start_{key}": value for key, value in start_geometry.items()})
        release_second = release_geometry.get("second_defender_distance_m", np.nan)
        start_second = start_geometry.get("second_defender_distance_m", np.nan)
        row["help_defender_closing_m"] = (
            float(start_second - release_second)
            if np.isfinite(start_second) and np.isfinite(release_second)
            else np.nan
        )
        row["density_change_within_3m"] = (
            float(
                release_geometry.get("defenders_within_3m", np.nan)
                - start_geometry.get("defenders_within_3m", np.nan)
            )
            if release_geometry and start_geometry
            else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)
