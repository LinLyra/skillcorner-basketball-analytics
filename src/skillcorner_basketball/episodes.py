from __future__ import annotations

import pandas as pd


def extract_frame_window(
    frames: pd.DataFrame,
    entities: pd.DataFrame,
    *,
    center_frame: int,
    frames_before: int,
    frames_after: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract an inclusive tracking window around an event frame."""
    start = int(center_frame) - int(frames_before)
    end = int(center_frame) + int(frames_after)
    frame_window = frames[frames["frame_idx"].between(start, end)].copy()
    entity_window = entities[entities["frame_idx"].between(start, end)].copy()
    return frame_window, entity_window


def extract_shot_window(
    shot: pd.Series | dict,
    frames: pd.DataFrame,
    entities: pd.DataFrame,
    *,
    seconds_before: float = 2.0,
    seconds_after: float = 0.5,
    fps: float = 25.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start_frame = int(shot["startFrame"])
    return extract_frame_window(
        frames,
        entities,
        center_frame=start_frame,
        frames_before=round(seconds_before * fps),
        frames_after=round(seconds_after * fps),
    )

