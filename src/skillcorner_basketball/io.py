from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

import pandas as pd

from .constants import ENTITY_COLUMNS, FEET_TO_METERS, FRAME_COLUMNS


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_tracking(path: str | Path) -> Iterator[dict]:
    """Yield raw tracking frames without loading the full game into memory."""
    source = Path(path)
    with _open_text(source) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {source}") from exc


def _xyz_metres(obj: dict) -> tuple[float | None, float | None, float | None]:
    xyz = obj.get("xyz")
    if not isinstance(xyz, list) or len(xyz) < 2:
        return None, None, None
    x = float(xyz[0]) * FEET_TO_METERS
    y = float(xyz[1]) * FEET_TO_METERS
    z = float(xyz[2]) * FEET_TO_METERS if len(xyz) > 2 and xyz[2] is not None else None
    return x, y, z


def _entity_row(
    obj: dict,
    *,
    game_id: int | str,
    frame_idx: int,
    wall_clock_ms: int,
    period: int,
    entity_type: str,
    team_side: str | None,
) -> dict:
    x_m, y_m, z_m = _xyz_metres(obj)
    speed = obj.get("speed")
    pred_error = obj.get("predError")
    return {
        "game_id": game_id,
        "frame_idx": frame_idx,
        "wall_clock_ms": wall_clock_ms,
        "period": period,
        "entity_type": entity_type,
        "team_side": team_side,
        "player_id": obj.get("playerId"),
        "jersey": obj.get("jersey"),
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        "speed_mps": float(speed) * FEET_TO_METERS if speed is not None else None,
        "is_detected": bool(obj["isDetected"]) if obj.get("isDetected") is not None else None,
        "pred_error_m": float(pred_error) * FEET_TO_METERS if pred_error is not None else None,
    }


def load_tracking(path: str | Path, game_id: int | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize nested SkillCorner tracking into frame and entity tables."""
    frame_rows: list[dict] = []
    entity_rows: list[dict] = []

    for raw in iter_tracking(path):
        frame_idx = int(raw["frameIdx"])
        wall_clock_ms = int(raw["wallClock"])
        period = int(raw["period"])
        frame_rows.append(
            {
                "game_id": game_id,
                "frame_idx": frame_idx,
                "wall_clock_ms": wall_clock_ms,
                "period": period,
                "game_clock_s": raw.get("gameClock"),
                "shot_clock_s": raw.get("shotClock"),
                "game_clock_stopped": raw.get("gameClockStopped"),
            }
        )

        for team_side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
            for player in raw.get(key) or []:
                entity_rows.append(
                    _entity_row(
                        player,
                        game_id=game_id,
                        frame_idx=frame_idx,
                        wall_clock_ms=wall_clock_ms,
                        period=period,
                        entity_type="player",
                        team_side=team_side,
                    )
                )

        ball = raw.get("ball") or {}
        if ball:
            entity_rows.append(
                _entity_row(
                    ball,
                    game_id=game_id,
                    frame_idx=frame_idx,
                    wall_clock_ms=wall_clock_ms,
                    period=period,
                    entity_type="ball",
                    team_side=None,
                )
            )

    frames = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    entities = pd.DataFrame(entity_rows, columns=ENTITY_COLUMNS)
    return frames, entities


def load_dynamic_events(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load the event bundle while keeping each event family as its own table."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Dynamic events file must contain one JSON object")
    return {
        name: pd.DataFrame(records if isinstance(records, list) else [])
        for name, records in payload.items()
    }

