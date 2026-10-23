from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skillcorner_basketball.modeling import MODEL_SPECS, leave_one_game_out_predictions, score_predictions


TEMPORAL_COLUMNS = [
    "recent_closing_speed_mps",
    "separation_gain_m",
    "pressure_duration_s",
    "mean_distance_m",
    "min_distance_m",
    "max_distance_m",
]


def mean_delta(data: pd.DataFrame) -> float:
    predictions = leave_one_game_out_predictions(data, specs=MODEL_SPECS[1:])
    scores = score_predictions(predictions)
    pivot = scores.pivot(index="game_id", columns="model", values="log_loss")
    return float((pivot["M2_temporal_defense"] - pivot["M1_static_defense"]).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Within-game temporal-feature permutation control")
    parser.add_argument("--features", type=Path, default=Path("outputs/study/shot_features_1s.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/negative_control"))
    parser.add_argument("--figure", type=Path, default=Path("reports/figures/negative_control.png"))
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.features)
    data = data[
        (data["status"] == "ok")
        & (~data["fouled"])
        & data["release_distance_aligned_5cm"].eq(True)
    ].copy()
    observed = mean_delta(data)
    rng = np.random.default_rng(args.seed)
    null = []
    for repetition in range(args.repetitions):
        permuted = data.copy()
        for _, indices in permuted.groupby("game_id").groups.items():
            source = np.asarray(list(indices))
            target = rng.permutation(source)
            permuted.loc[source, TEMPORAL_COLUMNS] = data.loc[target, TEMPORAL_COLUMNS].to_numpy()
        null.append(mean_delta(permuted))
        if (repetition + 1) % 25 == 0:
            print(f"Completed {repetition + 1}/{args.repetitions}", flush=True)

    null = np.asarray(null)
    result = {
        "observed_delta_log_loss": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_2.5pct": float(np.quantile(null, 0.025)),
        "null_97.5pct": float(np.quantile(null, 0.975)),
        "one_sided_fraction_null_as_good_or_better": float(np.mean(null <= observed)),
        "repetitions": args.repetitions,
        "seed": args.seed,
        "permutation": "temporal feature vectors jointly permuted within game",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame({"permuted_delta_log_loss": null}).to_csv(
        args.output_dir / "distribution.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    ax.hist(null, bins=24, color="#94A3B8", edgecolor="white")
    ax.axvline(observed, color="#E4572E", linewidth=3, label=f"Observed: {observed:.4f}")
    ax.axvline(0, color="#111827", linewidth=1, linestyle="--")
    ax.set_xlabel("Temporal − static mean held-out-game log loss")
    ax.set_ylabel("Permutation count")
    ax.set_title("Negative control: break temporal alignment within each game")
    ax.legend(frameon=False)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=220, bbox_inches="tight")
    fig.savefig(args.figure.with_suffix(".pdf"), bbox_inches="tight")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
