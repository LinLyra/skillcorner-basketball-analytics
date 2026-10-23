FEET_TO_METERS = 0.3048
TRACKING_FPS = 25.0

FRAME_COLUMNS = [
    "game_id",
    "frame_idx",
    "wall_clock_ms",
    "period",
    "game_clock_s",
    "shot_clock_s",
    "game_clock_stopped",
]

ENTITY_COLUMNS = [
    "game_id",
    "frame_idx",
    "wall_clock_ms",
    "period",
    "entity_type",
    "team_side",
    "player_id",
    "jersey",
    "x_m",
    "y_m",
    "z_m",
    "speed_mps",
    "is_detected",
    "pred_error_m",
]

