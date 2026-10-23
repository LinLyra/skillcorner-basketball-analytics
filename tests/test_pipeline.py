import gzip
import json

import numpy as np

from skillcorner_basketball.episodes import extract_shot_window
from skillcorner_basketball.features import build_shot_temporal_features
from skillcorner_basketball.geometry import closing_speed, distance, separation_gain
from skillcorner_basketball.io import load_tracking
from skillcorner_basketball.modeling import (
    leave_one_game_out_predictions,
    paired_game_bootstrap,
    score_predictions,
)
from skillcorner_basketball.quality import tracking_quality_report
from skillcorner_basketball.spatial import build_shot_spatial_features
from skillcorner_basketball.validation import (
    shot_validation_summary,
    validate_shot_defender_distance,
)


def _frame(frame_idx: int, empty: bool = False) -> dict:
    players = [] if empty else [
        {
            "xyz": [10.0, 5.0, 0.0],
            "speed": 6.0,
            "jersey": "7",
            "playerId": 101,
            "isDetected": 1,
            "predError": 1.0,
        }
    ]
    return {
        "frameIdx": frame_idx,
        "wallClock": frame_idx * 40,
        "gameClock": 600.0 - frame_idx / 25,
        "gameClockStopped": empty,
        "period": 1,
        "shotClock": 24.0,
        "homePlayers": players,
        "awayPlayers": [],
        "ball": {} if empty else {
            "xyz": [11.0, 5.0, 4.0],
            "speed": 9.0,
            "isDetected": 0.0,
            "predError": 2.0,
        },
    }


def test_tracking_normalization_and_quality(tmp_path):
    path = tmp_path / "tracking.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for frame in (_frame(100), _frame(101, empty=True)):
            handle.write(json.dumps(frame) + "\n")

    frames, entities = load_tracking(path, game_id=42)
    report = tracking_quality_report(frames, entities)

    assert len(frames) == 2
    assert len(entities) == 2
    assert np.isclose(entities.iloc[0]["x_m"], 3.048)
    assert entities.iloc[1]["entity_type"] == "ball"
    assert bool(entities.iloc[1]["is_detected"]) is False
    assert report["empty_tracking_frames"] == 1
    assert report["player_detected_rate"] == 1.0


def test_episode_and_geometry(tmp_path):
    path = tmp_path / "tracking.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for frame_idx in range(10):
            handle.write(json.dumps(_frame(frame_idx)) + "\n")
    frames, entities = load_tracking(path, game_id=42)

    frame_window, entity_window = extract_shot_window(
        {"startFrame": 5}, frames, entities, seconds_before=0.08, seconds_after=0.04
    )
    assert frame_window["frame_idx"].tolist() == [3, 4, 5, 6]
    assert entity_window["frame_idx"].nunique() == 4
    assert distance(np.array([0, 0]), np.array([3, 4])) == 5.0
    assert closing_speed(
        np.array([0, 0]), np.array([2, 0]), np.array([0, 0]), np.array([-1, 0])
    ) == 1.0
    assert np.isclose(separation_gain(1.0, 1.6), 0.6)


def test_shot_defender_distance_validation(tmp_path):
    path = tmp_path / "tracking.jsonl.gz"
    frame = _frame(25)
    frame["awayPlayers"] = [
        {
            "xyz": [13.0, 9.0, 0.0],
            "speed": 0.0,
            "jersey": "8",
            "playerId": 202,
            "isDetected": 1,
            "predError": 1.0,
        }
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(frame) + "\n")
    _, entities = load_tracking(path, game_id=42)
    shots = __import__("pandas").DataFrame(
        [
            {
                "id": "shot-1",
                "startFrame": 25,
                "shooterId": 101,
                "closestDefId": "202",
                "closestDefDist": 5.0,
            }
        ]
    )
    validation = validate_shot_defender_distance(shots, entities)
    summary = shot_validation_summary(validation)

    assert validation.iloc[0]["status"] == "ok"
    assert np.isclose(validation.iloc[0]["recomputed_distance_m"], 1.524)
    assert np.isclose(validation.iloc[0]["absolute_error_m"], 0.0)
    assert summary["validated_rate"] == 1.0


def test_temporal_features_fix_release_defender_and_measure_closing(tmp_path):
    path = tmp_path / "tracking.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for frame_idx in range(6):
            frame = _frame(frame_idx)
            frame["homePlayers"][0]["xyz"] = [0.0, 0.0, 0.0]
            frame["awayPlayers"] = [
                {
                    "xyz": [10.0 - frame_idx, 0.0, 0.0],
                    "speed": 25.0,
                    "jersey": "8",
                    "playerId": 202,
                    "isDetected": 1,
                    "predError": 0.5,
                }
            ]
            handle.write(json.dumps(frame) + "\n")
    _, entities = load_tracking(path, game_id=42)
    shots = __import__("pandas").DataFrame(
        [
            {
                "id": "shot-1",
                "startFrame": 5,
                "shooterId": 101,
                "closestDefId": 202,
                "closestDefDist": 5.0,
                "outcome": True,
                "fouled": False,
                "blocked": False,
                "three": False,
            }
        ]
    )
    features = build_shot_temporal_features(
        shots, entities, game_id=42, window_seconds=0.2, recent_seconds=0.12
    )
    row = features.iloc[0]

    assert row["status"] == "ok"
    assert row["observed_frame_count"] == 6
    assert np.isclose(row["release_distance_m"], 1.524)
    assert np.isclose(row["window_start_distance_m"], 3.048)
    assert np.isclose(row["net_closing_speed_mps"], 7.62)
    assert row["both_detected_rate"] == 1.0

    noisy_a = build_shot_temporal_features(
        shots, entities, game_id=42, window_seconds=0.2, recent_seconds=0.12,
        coordinate_noise_seed=7,
    )
    noisy_b = build_shot_temporal_features(
        shots, entities, game_id=42, window_seconds=0.2, recent_seconds=0.12,
        coordinate_noise_seed=7,
    )
    assert np.isclose(noisy_a.iloc[0]["release_distance_m"], noisy_b.iloc[0]["release_distance_m"])
    assert not np.isclose(noisy_a.iloc[0]["release_distance_m"], row["release_distance_m"])


def test_leave_one_game_out_predictions_are_out_of_sample():
    pd = __import__("pandas")
    rows = []
    for game_id in (1, 2, 3):
        for index, outcome in enumerate((0, 1, 0, 1)):
            rows.append(
                {
                    "game_id": game_id,
                    "shot_id": f"{game_id}-{index}",
                    "outcome": outcome,
                    "shot_distance_ft": 5 + index,
                    "shot_clock_s": 10 + index,
                    "three": index > 1,
                    "catch_and_shoot": index % 2 == 0,
                    "dribbles_before": index,
                    "region": "paint" if index < 2 else "arc",
                    "shot_type": "jumper",
                    "release_distance_m": 1 + 0.1 * index,
                    "recent_closing_speed_mps": 0.2 * index,
                    "separation_gain_m": 0.1 * index,
                    "pressure_duration_s": 0.04 * index,
                    "mean_distance_m": 1.1 + 0.1 * index,
                    "min_distance_m": 0.9 + 0.1 * index,
                    "max_distance_m": 1.3 + 0.1 * index,
                }
            )
    data = pd.DataFrame(rows)
    predictions = leave_one_game_out_predictions(data)
    scores = score_predictions(predictions)

    assert len(predictions) == len(data) * 3
    assert predictions.groupby(["model", "game_id"]).size().eq(4).all()
    assert scores["log_loss"].notna().all()


def test_frame_nearest_defender_can_switch_identity(tmp_path):
    path = tmp_path / "tracking.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for frame_idx in range(6):
            frame = _frame(frame_idx)
            frame["homePlayers"][0]["xyz"] = [0.0, 0.0, 0.0]
            frame["awayPlayers"] = [
                {"xyz": [2.0 + frame_idx, 0.0, 0.0], "playerId": 202, "isDetected": 1, "predError": 0.2},
                {"xyz": [8.0 - frame_idx, 0.0, 0.0], "playerId": 203, "isDetected": 1, "predError": 0.2},
            ]
            handle.write(json.dumps(frame) + "\n")
    _, entities = load_tracking(path, game_id=42)
    shots = __import__("pandas").DataFrame(
        [{"id": "shot-1", "startFrame": 5, "shooterId": 101, "closestDefId": 203,
          "closestDefDist": 3.0, "outcome": False}]
    )
    features = build_shot_temporal_features(
        shots, entities, game_id=42, window_seconds=0.2, defender_mode="frame_nearest"
    )

    assert features.iloc[0]["status"] == "ok"
    assert features.iloc[0]["unique_defenders_in_window"] == 2
    assert np.isclose(features.iloc[0]["release_distance_m"], 0.9144)


def test_paired_game_bootstrap_reports_direction():
    pd = __import__("pandas")
    scores = pd.DataFrame(
        [
            {"game_id": game, "model": model, "log_loss": value}
            for game, static, temporal in [(1, 0.7, 0.6), (2, 0.8, 0.7), (3, 0.6, 0.55)]
            for model, value in [("static", static), ("temporal", temporal)]
        ]
    )
    result = paired_game_bootstrap(
        scores, metric="log_loss", model_a="temporal", model_b="static", repetitions=1000
    )

    assert result["mean_difference"] < 0
    assert result["games_favoring_a"] == 3
    assert result["ci95_high"] < 0


def test_smoothing_keeps_raw_release_distance(tmp_path):
    path = tmp_path / "tracking.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for frame_idx in range(6):
            frame = _frame(frame_idx)
            frame["homePlayers"][0]["xyz"] = [0.0, 0.0, 0.0]
            frame["awayPlayers"] = [{"xyz": [10 - frame_idx, 0, 0], "playerId": 202,
                                     "isDetected": 1, "predError": 0.2}]
            handle.write(json.dumps(frame) + "\n")
    _, entities = load_tracking(path, game_id=42)
    shots = __import__("pandas").DataFrame(
        [{"id": "shot-1", "startFrame": 5, "shooterId": 101, "closestDefId": 202,
          "closestDefDist": 5.0, "outcome": False}]
    )
    raw = build_shot_temporal_features(shots, entities, game_id=42, window_seconds=.2)
    smooth = build_shot_temporal_features(
        shots, entities, game_id=42, window_seconds=.2, smoothing_frames=5
    )

    assert np.isclose(raw.iloc[0]["release_distance_m"], smooth.iloc[0]["release_distance_m"])
    assert raw.iloc[0]["window_start_distance_m"] != smooth.iloc[0]["window_start_distance_m"]


def test_multiplayer_spatial_geometry_counts_help_defense(tmp_path):
    path = tmp_path / "tracking.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for frame_idx in (0, 25):
            frame = _frame(frame_idx)
            frame["homePlayers"] = [
                {"xyz": [-30, 0, 0], "playerId": 101, "isDetected": 1, "predError": .2},
                {"xyz": [-20, 10, 0], "playerId": 102, "isDetected": 1, "predError": .2},
            ]
            frame["awayPlayers"] = [
                {"xyz": [-32, 0, 0], "playerId": 201, "isDetected": 1, "predError": .2},
                {"xyz": [-30, 6 if frame_idx == 0 else 3, 0], "playerId": 202, "isDetected": 1, "predError": .2},
                {"xyz": [-10, 10, 0], "playerId": 203, "isDetected": 1, "predError": .2},
            ]
            handle.write(json.dumps(frame) + "\n")
    _, entities = load_tracking(path, game_id=42)
    shots = __import__("pandas").DataFrame(
        [{"id": "shot-1", "startFrame": 25, "shooterId": 101}]
    )
    features = build_shot_spatial_features(shots, entities, game_id=42)
    row = features.iloc[0]

    assert row["spatial_status"] == "ok"
    assert np.isclose(row["release_nearest_defender_distance_m"], 0.6096)
    assert np.isclose(row["help_defender_closing_m"], 0.9144)
    assert row["release_defenders_within_1_5m"] == 2
