# Beyond the Snapshot

**When does temporal defensive context improve basketball shot quality?**

This open study tests a tempting tracking-data claim: that the defender's movement before release predicts shot outcome better than the release snapshot alone. It uses all ten complete 25 Hz professional games in SkillCorner Open Data Basketball, strict leave-one-game-out (LOGO) validation, small-cluster inference, position-error propagation, and possession-level case studies.

![Revision evidence summary](reports/figures/revision_summary.png)

## Result

Of 1,459 shots, 1,433 have usable shooter–defender trajectories and 1,211 non-fouled shots pass a 5 cm event/tracking alignment audit. Adding one second of closing speed, separation gain, pressure duration, and distance history changes mean LOGO log loss from 0.6646 to 0.6633. The improvement is small and uncertain: Δ = −0.00125, game-cluster 95% interval [−0.00586, 0.00316], exact two-sided sign-flip p = 0.613.

The original unconditional permutation looked favorable, but it broke the relationship between temporal variables and release distance. After cross-fitted residualization on static/context features, 15.5% of within-game conditional permutations perform at least as well as observed. Nested tuning and a monotonic nonlinear tree reach the same substantive conclusion: no stable incremental temporal value is detected in these ten games.

The most actionable finding is definitional. A frame-wise “nearest defender” changes identity during 36.0% of one-second windows. Across 1,209 common predictions, fixed-release and frame-nearest policies correlate at 0.988, yet 68 shots differ by more than five probability points. When identity switches, the mean absolute prediction difference is 2.56 points versus 0.75 without a switch.

![Defender-policy case studies](reports/figures/defender_policy_cases.png)

SkillCorner Shot Quality has mean log loss 0.6101, numerically stronger than the open models. It is treated only as an external production reference: its training size, features, and possible overlap with these games are unknown, so this is not a fair head-to-head model comparison.

## Why this matters

This is not a new proprietary-looking pressure score. It is a reproducible audit showing when tracking definitions change the basketball question being measured. The compatible interval suggests this experiment would likely have exposed an average improvement larger than roughly 0.0059 log-loss units, subject to the severe limitation of only ten game clusters.

The repository contributes:

- release-frame event/tracking alignment checks;
- fixed-release and frame-nearest defender trajectories;
- exact sign-flip and conditional-residual permutation inference;
- nested LOGO tuned logistic and monotonic tree benchmarks;
- Monte Carlo propagation of `predError` under a documented sensitivity assumption;
- replay-ready possession tables and three defender-policy case studies;
- a reusable `sc-basketball` command-line interface.

## Reproduce

Python 3.10+ is required. The raw files are intentionally not committed. Download the official data version pinned in `datasets.lock.json`, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
./scripts/reproduce_all.sh
```

The full run executes tests, rebuilds all cohorts and models, performs 200 conditional permutations and 30 position-error Monte Carlo draws, renders figures, and validates release invariants. For a fast software check:

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
```

## Study map

- [Full report](reports/final_report.md)
- [Cup abstract and 60-second script](reports/analytics_cup_submission.md)
- [Robustness results](reports/robustness_results.md)
- [Data contract](docs/data_contract.md)
- [Literature gap matrix](docs/literature_gap_matrix.md)
- [Evaluation protocol](docs/evaluation_protocol.md)
- [NBA portfolio brief](reports/nba_portfolio_brief.md)

## Interpretation and limits

The result does **not** show that defensive motion is irrelevant. It shows that these simple summaries, with this public sample, do not earn a stable incremental-performance claim. Binary outcomes are noisy; only ten full games are public; broadcast-derived tracking contains extrapolated positions; orientation, contest height, tactical assignment, screens, and player effects are incompletely observed. Subgroup analyses are exploratory.

`predError` is documented as a 90% radial bound, not a Gaussian standard deviation. Monte Carlo sensitivity explicitly assumes independent isotropic Gaussian x/y error and converts the radius using `sigma = predError / sqrt(-2 log(0.1))`. It is an assumption audit, not a claim about SkillCorner's proprietary error distribution.

## License and attribution

Code is released under the [MIT License](LICENSE.md). Data remains governed by SkillCorner's terms and is not redistributed. This independent project is not affiliated with or endorsed by SkillCorner, PySport, the ACB, or any team.
