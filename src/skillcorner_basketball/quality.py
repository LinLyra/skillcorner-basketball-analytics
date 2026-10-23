from __future__ import annotations

import math

import pandas as pd


def tracking_quality_report(frames: pd.DataFrame, entities: pd.DataFrame) -> dict:
    player_rows = entities[entities["entity_type"] == "player"]
    ball_rows = entities[entities["entity_type"] == "ball"]
    counts = player_rows.groupby("frame_idx").size().reindex(frames["frame_idx"], fill_value=0)

    def safe_mean(series: pd.Series) -> float | None:
        value = float(series.dropna().mean()) if series.notna().any() else math.nan
        return None if math.isnan(value) else value

    player_error = player_rows["pred_error_m"].dropna()
    ball_error = ball_rows["pred_error_m"].dropna()
    return {
        "frame_count": int(len(frames)),
        "empty_tracking_frames": int((counts == 0).sum()),
        "frames_with_10_players": int((counts == 10).sum()),
        "mean_players_per_frame": safe_mean(counts),
        "player_rows": int(len(player_rows)),
        "ball_rows": int(len(ball_rows)),
        "player_detected_rate": safe_mean(player_rows["is_detected"].astype("Float64")),
        "ball_detected_rate": safe_mean(ball_rows["is_detected"].astype("Float64")),
        "mean_player_pred_error_m": safe_mean(player_rows["pred_error_m"]),
        "mean_ball_pred_error_m": safe_mean(ball_rows["pred_error_m"]),
        "p95_player_pred_error_m": float(player_error.quantile(0.95)) if len(player_error) else None,
        "p95_ball_pred_error_m": float(ball_error.quantile(0.95)) if len(ball_error) else None,
        "duplicate_frame_rows": int(frames.duplicated(["game_id", "frame_idx"]).sum()),
        "duplicate_entity_rows": int(
            entities.duplicated(["game_id", "frame_idx", "entity_type", "player_id"]).sum()
        ),
    }

