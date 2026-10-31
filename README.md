# Beyond the Snapshot

**When does temporal defensive context improve basketball shot quality?**

An independent, reproducible study of SkillCorner Open Data Basketball. It tests whether the defender's movement before release predicts shot outcome better than a release-time snapshot, and turns the measurement issues uncovered along the way into a reusable tracking-data audit.

The project uses all ten complete 25 Hz professional games, strict leave-one-game-out (LOGO) validation, exact small-cluster inference, position-error propagation, and possession-level case studies.

## At a glance

- **Question:** Do simple pre-release defender-motion summaries add predictive information beyond shot context and release distance?
- **Result:** The observed one-second gain is small (`Δ log loss = -0.00125`) and cannot be separated from zero with ten games.
- **Boundary:** The compatible range still includes an operationally relevant improvement; this is an inconclusive result, not evidence that motion is useless.
- **Actionable finding:** Frame-wise nearest-defender identity changes in 36.0% of one-second windows, making the defender definition consequential at possession level.
- **Deliverable:** An importable API and CLI for release alignment, defender-policy comparison, switching detection, and tracking-uncertainty audits.

![Revision evidence summary](reports/figures/revision_summary.png)

## Main result

Of 1,459 shots, 1,433 have usable shooter–defender trajectories and 1,211 non-fouled shots pass a 5 cm event/tracking alignment audit. Adding one second of closing speed, separation gain, pressure duration, and distance history changes mean LOGO log loss from 0.6646 to 0.6633. The improvement is small and uncertain: Δ = −0.00125, game-cluster 95% interval [−0.00586, 0.00316], exact two-sided sign-flip p = 0.613.

The original unconditional permutation looked favorable, but it broke the relationship between temporal variables and release distance. After cross-fitted residualization on static/context features, 15.5% of within-game conditional permutations perform at least as well as observed. Nested tuning and a monotonic nonlinear tree reach the same substantive conclusion: this sample cannot resolve the small observed increment from zero.

The most actionable analysis is definitional. A frame-wise “nearest defender” changes identity during 36.0% of one-second windows. This is a quantified sensitivity—not a claim that practitioners are unaware of switching, and not evidence that switching causes prediction differences. The study also tests a 0.3 m hysteresis policy and compares all policies on an identical held-out cohort.

![Defender-policy case studies](reports/figures/defender_policy_cases.png)

SkillCorner Shot Quality has mean log loss 0.6101, numerically stronger than the open models. It is treated only as an external production reference: its training size, features, and possible overlap with these games are unknown, so this is not a fair head-to-head model comparison.

## Why this matters

Tracking features are only useful if event alignment, player identity, and uncertainty are handled explicitly. This repository shows how those choices affect the measured construct before making a performance claim.

Inverting the exact sign-flip test gives a 95% compatible location interval of [−0.00690, 0.00428] for M2−M1 log loss. The study is therefore inconclusive at a retrospective operational reference of 0.005: it cannot rule out an improvement as large as roughly 0.0069. This interval is not a formal minimum-detectable-effect calculation.

The repository contributes:

- release-frame event/tracking alignment checks;
- fixed-release and frame-nearest defender trajectories;
- exact sign-flip and conditional-residual permutation inference;
- nested LOGO tuned logistic and monotonic tree benchmarks;
- Monte Carlo propagation of `predError` under a documented sensitivity assumption;
- replay-ready possession tables and three defender-policy case studies;
- an importable `audit_game()` API plus the `sc-basketball` command-line interface.

## Reviewer quick path

1. Read the [full report](reports/final_report.md) for the research design and interpretation.
2. Open the [robustness report](reports/robustness_results.md) for defender-policy and inference checks.
3. Review the compact [Analytics Cup abstract and presentation script](reports/analytics_cup_submission.md).
4. Run `pytest -q` and `python scripts/validate_release.py` to verify the software and committed result invariants.

## Install and reproduce

Python 3.10+ is required. The raw files are intentionally not committed. Download the official data version pinned in `datasets.lock.json`, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
for game_id in 114086 114099 114169 114234 114243 178442 179612 184439 188630 191313; do
  python scripts/fetch_official_game.py --game-id "$game_id"
done
./scripts/reproduce_all.sh
```

The fetch script downloads files only from the pinned upstream SkillCorner commit and verifies their byte sizes and SHA-256 hashes. The full run executes tests, rebuilds all cohorts and models, performs 200 conditional permutations and 200 position-error Monte Carlo draws, renders figures, and validates release invariants. For a fast software check that does not require raw data:

```bash
pytest -q
python scripts/validate_release.py
```

Reusable CLI examples:

```bash
sc-basketball validate-shots TRACKING.jsonl.gz EVENTS.json \
  --game-id GAME_ID --output-dir audit

sc-basketball build-shot-features TRACKING.jsonl.gz EVENTS.json \
  --game-id GAME_ID --window-seconds 1 \
  --defender-mode frame_nearest --output features.csv

sc-basketball audit-game TRACKING.jsonl.gz EVENTS.json \
  --game-id GAME_ID --output-dir audit_bundle
```

The same audit is importable without project scripts:

```python
from skillcorner_basketball import AuditConfig, audit_game

audit = audit_game(
    "tracking.jsonl.gz", "dynamic_events.json", game_id="GAME_ID",
    config=AuditConfig(window_seconds=1.0, defender_switch_margin_m=0.3),
)
audit.write("audit_bundle")
```

## Study map

- [Full report](reports/final_report.md)
- [Cup abstract and 60-second script](reports/analytics_cup_submission.md)
- [Robustness results](reports/robustness_results.md)
- [Data contract](docs/data_contract.md)
- [Literature gap matrix](docs/literature_gap_matrix.md)
- [Evaluation protocol](docs/evaluation_protocol.md)

## Interpretation and limits

The result does **not** show that defensive motion is irrelevant. It shows that these simple summaries, with this public sample, do not earn a stable incremental-performance claim. Binary outcomes are noisy; only ten full games are public; broadcast-derived tracking contains extrapolated positions; orientation, contest height, tactical assignment, screens, and player effects are incompletely observed. Subgroup analyses are exploratory.

`predError` is documented as a 90% radial bound, not a Gaussian standard deviation. Monte Carlo sensitivity explicitly assumes independent isotropic Gaussian x/y error and converts the radius using `sigma = predError / sqrt(-2 log(0.1))`. It is an assumption audit, not a claim about SkillCorner's proprietary error distribution.

## License and attribution

Code is released under the [MIT License](LICENSE). Data remains governed by SkillCorner's terms and is not redistributed. This independent project is not affiliated with or endorsed by SkillCorner, PySport, the ACB, or any team.
