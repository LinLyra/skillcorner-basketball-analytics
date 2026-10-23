from __future__ import annotations

import numpy as np


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance in the units supplied by the caller."""
    return float(np.linalg.norm(np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)))


def closing_speed(
    offense_position: np.ndarray,
    defense_position: np.ndarray,
    offense_velocity: np.ndarray,
    defense_velocity: np.ndarray,
) -> float:
    """Rate at which separation is shrinking; positive means defender is closing."""
    offense_position = np.asarray(offense_position, dtype=float)
    defense_position = np.asarray(defense_position, dtype=float)
    delta = defense_position - offense_position
    separation = np.linalg.norm(delta)
    if separation == 0:
        return 0.0
    unit_offense_to_defense = delta / separation
    relative_velocity = np.asarray(defense_velocity, dtype=float) - np.asarray(
        offense_velocity, dtype=float
    )
    return float(-np.dot(relative_velocity, unit_offense_to_defense))


def separation_gain(before: float, after: float) -> float:
    return float(after - before)

