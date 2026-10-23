from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...] = ("region", "shot_type")


MODEL_SPECS = (
    ModelSpec(
        "M0_location_context",
        ("shot_distance_ft", "shot_clock_s", "three", "catch_and_shoot", "dribbles_before"),
    ),
    ModelSpec(
        "M1_static_defense",
        (
            "shot_distance_ft",
            "shot_clock_s",
            "three",
            "catch_and_shoot",
            "dribbles_before",
            "release_distance_m",
        ),
    ),
    ModelSpec(
        "M2_temporal_defense",
        (
            "shot_distance_ft",
            "shot_clock_s",
            "three",
            "catch_and_shoot",
            "dribbles_before",
            "release_distance_m",
            "recent_closing_speed_mps",
            "separation_gain_m",
            "pressure_duration_s",
            "mean_distance_m",
            "min_distance_m",
            "max_distance_m",
        ),
    ),
)

SPATIAL_MODEL_SPEC = ModelSpec(
    "M3_multiplayer_geometry",
    MODEL_SPECS[-1].numeric
    + (
        "release_second_defender_distance_m",
        "release_third_defender_distance_m",
        "release_defenders_within_1_5m",
        "release_defenders_within_3m",
        "release_nearest_defender_angle_to_hoop_deg",
        "release_defenders_in_30deg_cone",
        "release_min_lane_obstruction_m",
        "release_defensive_centroid_distance_m",
        "release_nearest_teammate_distance_m",
        "release_mean_teammate_distance_m",
        "help_defender_closing_m",
        "density_change_within_3m",
    ),
)


def _pipeline(spec: ModelSpec) -> Pipeline:
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = [("numeric", numeric, list(spec.numeric))]
    if spec.categorical:
        transformers.append(("categorical", categorical, list(spec.categorical)))
    preprocessor = ColumnTransformer(transformers)
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LogisticRegression(C=1.0, max_iter=2000, solver="liblinear")),
        ]
    )


def leave_one_game_out_predictions(
    data: pd.DataFrame, specs: tuple[ModelSpec, ...] = MODEL_SPECS
) -> pd.DataFrame:
    """Generate strictly out-of-game probabilities for fixed model specifications."""
    required = {"game_id", "shot_id", "outcome"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"analysis table missing required columns: {missing}")
    rows: list[pd.DataFrame] = []
    for held_out_game in sorted(data["game_id"].unique()):
        train = data[data["game_id"] != held_out_game]
        test = data[data["game_id"] == held_out_game]
        if train["outcome"].nunique() < 2 or test.empty:
            continue
        for spec in specs:
            columns = list(spec.numeric + spec.categorical)
            pipeline = _pipeline(spec)
            pipeline.fit(train[columns], train["outcome"].astype(int))
            probability = pipeline.predict_proba(test[columns])[:, 1]
            rows.append(
                pd.DataFrame(
                    {
                        "game_id": test["game_id"].to_numpy(),
                        "shot_id": test["shot_id"].to_numpy(),
                        "outcome": test["outcome"].astype(int).to_numpy(),
                        "model": spec.name,
                        "probability": probability,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (model, game_id), group in predictions.groupby(["model", "game_id"], sort=True):
        y = group["outcome"].to_numpy()
        p = np.clip(group["probability"].to_numpy(), 1e-6, 1 - 1e-6)
        rows.append(
            {
                "model": model,
                "game_id": game_id,
                "n": len(group),
                "made_rate": float(np.mean(y)),
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                "calibration_in_large": float(np.mean(p) - np.mean(y)),
            }
        )
    return pd.DataFrame(rows)


def paired_game_bootstrap(
    scores: pd.DataFrame,
    *,
    metric: str,
    model_a: str,
    model_b: str,
    repetitions: int = 20_000,
    seed: int = 20261024,
) -> dict:
    """Cluster bootstrap a paired model difference by held-out game.

    The reported difference is model_a minus model_b. For loss metrics,
    negative values favor model_a.
    """
    pivot = scores.pivot(index="game_id", columns="model", values=metric).dropna(
        subset=[model_a, model_b]
    )
    differences = (pivot[model_a] - pivot[model_b]).to_numpy(dtype=float)
    if len(differences) < 2:
        raise ValueError("At least two paired games are required")
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(repetitions, len(differences)), replace=True).mean(axis=1)
    return {
        "metric": metric,
        "model_a": model_a,
        "model_b": model_b,
        "game_count": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "games_favoring_a": int(np.sum(differences < 0)) if metric in {"log_loss", "brier"} else int(np.sum(differences > 0)),
    }
