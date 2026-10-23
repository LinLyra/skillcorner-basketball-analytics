from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import FEET_TO_METERS


def _normalise_id(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def validate_shot_defender_distance(
    shots: pd.DataFrame,
    entities: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute official closest-defender distance at each shot release frame.

    The output is an integration check for event/tracking alignment and unit
    conversion. It is not a replacement for SkillCorner's shot definition.
    """
    required_shots = {"id", "startFrame", "shooterId", "closestDefId", "closestDefDist"}
    missing = sorted(required_shots - set(shots.columns))
    if missing:
        raise ValueError(f"shots table missing required columns: {missing}")

    players = entities[entities["entity_type"] == "player"].copy()
    players["player_key"] = players["player_id"].map(_normalise_id)
    by_frame = {int(frame): group for frame, group in players.groupby("frame_idx", sort=False)}

    rows: list[dict] = []
    for shot in shots.to_dict("records"):
        frame_idx = int(shot["startFrame"])
        frame_players = by_frame.get(frame_idx)
        shooter_key = _normalise_id(shot.get("shooterId"))
        defender_key = _normalise_id(shot.get("closestDefId"))
        status = "ok"
        recomputed_m = np.nan
        shooter_detected = pd.NA
        defender_detected = pd.NA
        combined_pred_error_m = np.nan

        if frame_players is None:
            status = "missing_frame"
        else:
            shooter = frame_players[frame_players["player_key"] == shooter_key]
            defender = frame_players[frame_players["player_key"] == defender_key]
            if shooter.empty:
                status = "missing_shooter"
            elif defender.empty:
                status = "missing_defender"
            else:
                shooter_row = shooter.iloc[0]
                defender_row = defender.iloc[0]
                recomputed_m = float(
                    np.hypot(
                        defender_row["x_m"] - shooter_row["x_m"],
                        defender_row["y_m"] - shooter_row["y_m"],
                    )
                )
                shooter_detected = shooter_row["is_detected"]
                defender_detected = defender_row["is_detected"]
                combined_pred_error_m = float(
                    (shooter_row["pred_error_m"] or 0.0)
                    + (defender_row["pred_error_m"] or 0.0)
                )

        official_m = (
            float(shot["closestDefDist"]) * FEET_TO_METERS
            if shot.get("closestDefDist") is not None and not pd.isna(shot.get("closestDefDist"))
            else np.nan
        )
        rows.append(
            {
                "shot_id": shot["id"],
                "frame_idx": frame_idx,
                "shooter_id": shooter_key,
                "closest_defender_id": defender_key,
                "status": status,
                "official_distance_m": official_m,
                "recomputed_distance_m": recomputed_m,
                "absolute_error_m": abs(recomputed_m - official_m),
                "shooter_detected": shooter_detected,
                "defender_detected": defender_detected,
                "combined_pred_error_m": combined_pred_error_m,
            }
        )
    return pd.DataFrame(rows)


def shot_validation_summary(validation: pd.DataFrame) -> dict:
    ok = validation[validation["status"] == "ok"]
    errors = ok["absolute_error_m"].dropna()
    return {
        "shot_count": int(len(validation)),
        "validated_count": int(len(ok)),
        "validated_rate": float(len(ok) / len(validation)) if len(validation) else None,
        "median_absolute_error_m": float(errors.median()) if len(errors) else None,
        "p95_absolute_error_m": float(errors.quantile(0.95)) if len(errors) else None,
        "status_counts": {str(k): int(v) for k, v in validation["status"].value_counts().items()},
    }

