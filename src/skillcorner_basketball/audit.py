from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from .features import build_shot_temporal_features
from .io import load_dynamic_events, load_tracking
from .quality import tracking_quality_report
from .validation import shot_validation_summary, validate_shot_defender_distance


@dataclass(frozen=True)
class AuditConfig:
    window_seconds: float = 1.0
    alignment_tolerance_m: float = 0.05
    defender_switch_margin_m: float = 0.3
    fps: float = 25.0


@dataclass
class GameAudit:
    game_id: str
    tracking_quality: dict
    release_validation: pd.DataFrame
    policy_features: dict[str, pd.DataFrame]
    summary: dict

    def write(self, output_dir: str | Path) -> None:
        """Write a portable audit bundle without requiring project scripts."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.release_validation.to_csv(target / "release_validation.csv", index=False)
        for policy, table in self.policy_features.items():
            table.to_csv(target / f"shot_features_{policy}.csv", index=False)
        (target / "audit_summary.json").write_text(
            json.dumps(self.summary, indent=2) + "\n", encoding="utf-8"
        )


def audit_game(
    tracking_path: str | Path,
    events_path: str | Path,
    *,
    game_id: int | str,
    config: AuditConfig | None = None,
) -> GameAudit:
    """Run release-alignment and defender-policy audits for one game.

    This is the public high-level API. It loads the official files, validates
    release distances, constructs three documented defender policies, and
    returns both row-level evidence and a JSON-serializable summary.
    """
    cfg = config or AuditConfig()
    frames, entities = load_tracking(tracking_path, game_id)
    events = load_dynamic_events(events_path)
    if "shots" not in events:
        raise ValueError("Dynamic events bundle does not contain a shots table")
    shots = events["shots"]
    validation = validate_shot_defender_distance(shots, entities)
    policies = {
        "fixed_release": {},
        "frame_nearest": {},
        "frame_nearest_hysteresis": {
            "defender_switch_margin_m": cfg.defender_switch_margin_m
        },
    }
    features = {
        policy: build_shot_temporal_features(
            shots,
            entities,
            game_id=game_id,
            window_seconds=cfg.window_seconds,
            fps=cfg.fps,
            defender_mode=policy,
            **kwargs,
        )
        for policy, kwargs in policies.items()
    }
    policy_summary = {}
    for policy, table in features.items():
        valid = table[table["status"] == "ok"]
        policy_summary[policy] = {
            "valid_trajectories": int(len(valid)),
            "identity_switch_count": int((valid["unique_defenders_in_window"] > 1).sum()),
            "identity_switch_rate": float((valid["unique_defenders_in_window"] > 1).mean())
            if len(valid) else None,
            "aligned_count": int(
                (valid["release_distance_error_m"] <= cfg.alignment_tolerance_m).sum()
            ),
        }
    summary = {
        "game_id": str(game_id),
        "config": {
            "window_seconds": cfg.window_seconds,
            "alignment_tolerance_m": cfg.alignment_tolerance_m,
            "defender_switch_margin_m": cfg.defender_switch_margin_m,
            "fps": cfg.fps,
        },
        "tracking_quality": tracking_quality_report(frames, entities),
        "release_validation": shot_validation_summary(validation),
        "defender_policies": policy_summary,
    }
    return GameAudit(str(game_id), summary["tracking_quality"], validation, features, summary)
