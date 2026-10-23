from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from skillcorner_basketball.modeling import MODEL_SPECS, leave_one_game_out_predictions, score_predictions


TEMPORAL = [
    "recent_closing_speed_mps", "separation_gain_m", "pressure_duration_s",
    "mean_distance_m", "min_distance_m", "max_distance_m",
]
STATIC_NUMERIC = [
    "shot_distance_ft", "shot_clock_s", "three", "catch_and_shoot",
    "dribbles_before", "release_distance_m",
]
STATIC_CATEGORICAL = ["region", "shot_type"]


def model_delta(data: pd.DataFrame) -> tuple[float, pd.Series]:
    predictions = leave_one_game_out_predictions(data, specs=MODEL_SPECS[1:])
    scores = score_predictions(predictions)
    pivot = scores.pivot(index="game_id", columns="model", values="log_loss")
    differences = pivot["M2_temporal_defense"] - pivot["M1_static_defense"]
    return float(differences.mean()), differences


def cross_fitted_expectations(data: pd.DataFrame) -> np.ndarray:
    expected = np.full((len(data), len(TEMPORAL)), np.nan)
    for held_out in sorted(data["game_id"].unique()):
        train_mask = data["game_id"] != held_out
        test_mask = ~train_mask
        numeric = Pipeline(
            [("impute", SimpleImputer(strategy="median")),
             ("poly", PolynomialFeatures(degree=2, include_bias=False)),
             ("scale", StandardScaler())]
        )
        categorical = Pipeline(
            [("impute", SimpleImputer(strategy="most_frequent")),
             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )
        prep = ColumnTransformer(
            [("numeric", numeric, STATIC_NUMERIC), ("categorical", categorical, STATIC_CATEGORICAL)]
        )
        model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=10.0))])
        model.fit(data.loc[train_mask], data.loc[train_mask, TEMPORAL])
        expected[np.flatnonzero(test_mask)] = model.predict(data.loc[test_mask])
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description="Small-cluster and conditional-permutation checks")
    parser.add_argument("--features", type=Path, default=Path("outputs/study/shot_features_1s.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference_checks"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/inference_checks.png"))
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.features)
    data = data[
        (data["status"] == "ok") & (~data["fouled"])
        & data["release_distance_aligned_5cm"].eq(True)
    ].reset_index(drop=True)
    observed, game_differences = model_delta(data)

    sign_means = np.array([
        np.mean(game_differences.to_numpy() * np.asarray(signs))
        for signs in itertools.product((-1, 1), repeat=len(game_differences))
    ])
    sign_p_two_sided = float(np.mean(np.abs(sign_means) >= abs(observed) - 1e-15))
    sign_p_one_sided = float(np.mean(sign_means <= observed + 1e-15))

    expected = cross_fitted_expectations(data)
    residuals = data[TEMPORAL].to_numpy(dtype=float) - expected
    rng = np.random.default_rng(args.seed)
    conditional_null = []
    for repetition in range(args.repetitions):
        permuted = data.copy()
        rebuilt = expected.copy()
        for _, indices in data.groupby("game_id").groups.items():
            indices = np.asarray(list(indices))
            rebuilt[indices] += residuals[rng.permutation(indices)]
        permuted[TEMPORAL] = rebuilt
        delta, _ = model_delta(permuted)
        conditional_null.append(delta)
        if (repetition + 1) % 25 == 0:
            print(f"Conditional permutations: {repetition + 1}/{args.repetitions}", flush=True)
    conditional_null = np.asarray(conditional_null)
    conditional_fraction = float(np.mean(conditional_null <= observed))

    summary = {
        "observed_delta_log_loss": observed,
        "exact_sign_flip_two_sided_p": sign_p_two_sided,
        "exact_sign_flip_one_sided_p": sign_p_one_sided,
        "sign_flip_permutations": int(len(sign_means)),
        "conditional_null_mean": float(conditional_null.mean()),
        "conditional_null_2.5pct": float(np.quantile(conditional_null, 0.025)),
        "conditional_null_97.5pct": float(np.quantile(conditional_null, 0.975)),
        "conditional_fraction_as_good_or_better": conditional_fraction,
        "conditional_repetitions": args.repetitions,
        "residualization": "cross-fitted quadratic Ridge on static/context features; residual vectors permuted jointly within game",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame({"game_id": game_differences.index, "delta_log_loss": game_differences.values}).to_csv(
        args.output_dir / "game_differences.csv", index=False
    )
    pd.DataFrame({"conditional_delta_log_loss": conditional_null}).to_csv(
        args.output_dir / "conditional_null.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    ordered = game_differences.sort_values()
    axes[0].barh(np.arange(len(ordered)), ordered, color=np.where(ordered < 0, "#2A9D8F", "#D1495B"))
    axes[0].axvline(0, color="#111827", linewidth=1)
    axes[0].set_yticks(np.arange(len(ordered)), ordered.index.astype(str))
    axes[0].set_xlabel("Temporal − static log loss")
    axes[0].set_ylabel("Held-out game")
    axes[0].set_title(f"Exact sign-flip p={sign_p_two_sided:.3f} (two-sided)")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].hist(conditional_null, bins=24, color="#94A3B8", edgecolor="white")
    axes[1].axvline(observed, color="#E4572E", linewidth=3, label=f"Observed {observed:.4f}")
    axes[1].axvline(0, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Temporal − static mean game log loss")
    axes[1].set_ylabel("Conditional permutation count")
    axes[1].set_title(f"Residual permutation fraction ≤ observed: {conditional_fraction:.3f}")
    axes[1].legend(frameon=False)
    fig.suptitle("Small-cluster inference and conditional negative control", fontsize=14)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=230, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
