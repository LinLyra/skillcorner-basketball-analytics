from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

from skillcorner_basketball.modeling import (
    MODEL_SPECS,
    _pipeline,
    paired_game_bootstrap,
    score_predictions,
)


LOGISTIC_GRID = [0.01, 0.1, 1.0, 10.0]
TREE_GRID = [
    {"max_leaf_nodes": 3, "l2_regularization": 1.0},
    {"max_leaf_nodes": 3, "l2_regularization": 10.0},
    {"max_leaf_nodes": 7, "l2_regularization": 1.0},
    {"max_leaf_nodes": 7, "l2_regularization": 10.0},
]


def inner_logo_logistic(train: pd.DataFrame, spec) -> float:
    candidates = []
    columns = list(spec.numeric + spec.categorical)
    for c in LOGISTIC_GRID:
        losses = []
        for held_out in sorted(train["game_id"].unique()):
            inner_train = train[train["game_id"] != held_out]
            inner_test = train[train["game_id"] == held_out]
            model = _pipeline(spec)
            model.set_params(model__C=c)
            model.fit(inner_train[columns], inner_train["outcome"].astype(int))
            probability = model.predict_proba(inner_test[columns])[:, 1]
            losses.append(log_loss(inner_test["outcome"], probability, labels=[0, 1]))
        candidates.append((float(np.mean(losses)), c))
    return min(candidates)[1]


def fit_tree(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], params: dict, monotonic: list[int]):
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(train[columns])
    x_test = imputer.transform(test[columns])
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"],
        monotonic_cst=monotonic,
        early_stopping=False,
        random_state=20261024,
    )
    model.fit(x_train, train["outcome"].astype(int))
    return model.predict_proba(x_test)[:, 1]


def inner_logo_tree(train: pd.DataFrame, columns: list[str], monotonic: list[int]) -> dict:
    candidates = []
    for params in TREE_GRID:
        losses = []
        for held_out in sorted(train["game_id"].unique()):
            inner_train = train[train["game_id"] != held_out]
            inner_test = train[train["game_id"] == held_out]
            probability = fit_tree(inner_train, inner_test, columns, params, monotonic)
            losses.append(log_loss(inner_test["outcome"], probability, labels=[0, 1]))
        candidates.append((float(np.mean(losses)), params))
    return min(candidates, key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested LOGO tuned linear and monotonic tree models")
    parser.add_argument("--features", type=Path, default=Path("outputs/study/shot_features_1s.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nested_models"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.features)
    data = data[
        (data["status"] == "ok") & (~data["fouled"])
        & data["release_distance_aligned_5cm"].eq(True)
    ].copy()
    static_spec, temporal_spec = MODEL_SPECS[1], MODEL_SPECS[2]
    static_columns = list(static_spec.numeric)
    temporal_columns = list(temporal_spec.numeric)
    static_monotonic = [-1, 0, 0, 0, 0, 1]
    temporal_monotonic = static_monotonic + [-1, 1, -1, 1, 1, 1]
    prediction_rows = []
    selected_rows = []

    for held_out in sorted(data["game_id"].unique()):
        print(f"Outer held-out game {held_out}", flush=True)
        train = data[data["game_id"] != held_out]
        test = data[data["game_id"] == held_out]
        for name, spec in (("TunedLogit_static", static_spec), ("TunedLogit_temporal", temporal_spec)):
            best_c = inner_logo_logistic(train, spec)
            columns = list(spec.numeric + spec.categorical)
            model = _pipeline(spec)
            model.set_params(model__C=best_c)
            model.fit(train[columns], train["outcome"].astype(int))
            probability = model.predict_proba(test[columns])[:, 1]
            selected_rows.append({"game_id": held_out, "model": name, "C": best_c})
            prediction_rows.append(pd.DataFrame({
                "game_id": test["game_id"], "shot_id": test["shot_id"],
                "outcome": test["outcome"].astype(int), "model": name, "probability": probability,
            }))
        for name, columns, monotonic in (
            ("MonotonicTree_static", static_columns, static_monotonic),
            ("MonotonicTree_temporal", temporal_columns, temporal_monotonic),
        ):
            params = inner_logo_tree(train, columns, monotonic)
            probability = fit_tree(train, test, columns, params, monotonic)
            selected_rows.append({"game_id": held_out, "model": name, **params})
            prediction_rows.append(pd.DataFrame({
                "game_id": test["game_id"], "shot_id": test["shot_id"],
                "outcome": test["outcome"].astype(int), "model": name, "probability": probability,
            }))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    scores = score_predictions(predictions)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    scores.to_csv(args.output_dir / "scores.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(args.output_dir / "selected_hyperparameters.csv", index=False)
    comparisons = []
    for temporal, static in (
        ("TunedLogit_temporal", "TunedLogit_static"),
        ("MonotonicTree_temporal", "MonotonicTree_static"),
    ):
        for metric in ("log_loss", "brier", "auc"):
            comparisons.append(
                paired_game_bootstrap(scores, metric=metric, model_a=temporal, model_b=static)
            )
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    summary = scores.groupby("model")[["log_loss", "brier", "auc"]].agg(["mean", "std"])
    summary.to_csv(args.output_dir / "summary.csv")
    print(summary.to_string(), flush=True)
    print(comparisons.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
