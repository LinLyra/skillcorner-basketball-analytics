# Analytics Cup submission package

## Title

**Beyond the Snapshot: When Does Temporal Defensive Context Improve Basketball Shot Quality?**

## One-sentence contribution

A reproducible audit shows that simple pre-release motion summaries do not deliver stable incremental value beyond release distance, and that defender identity choices can materially change possession-level conclusions.

## Short abstract

Tracking data invites increasingly complex descriptions of defensive pressure, but more motion features do not necessarily generalize. Using ten SkillCorner professional basketball games at 25 Hz, this project compares location/context, release-distance, temporal-defense, and multiplayer-geometry models under leave-one-game-out validation. A one-second temporal model changes mean log loss by −0.00125; exact sign-flip p = 0.613 and the inverted compatible interval is [−0.00690, 0.00428]. The design is therefore inconclusive at a retrospective practical reference of 0.005, rather than evidence of no effect. A cross-fitted conditional permutation, nested tuning, and a monotonic tree support the same interpretation. Frame-wise nearest-defender identity changes in 36% of trajectories; the work quantifies this known definitional sensitivity, tests a 0.3 m hysteresis policy, and compares policies on identical held-out shots without claiming switching causes prediction differences. The project contributes an importable audit API, explicit position-error propagation, and possession-level visual explanations. SkillCorner Shot Quality is shown only as an external production reference because its training provenance and overlap are unknown.

## 60-second presentation script

> Shot quality is often judged from one frame. I asked whether the previous second adds predictive value. Across 1,459 shots and strict leave-one-game-out validation, the exact compatible interval ran from a 0.0069 log-loss improvement to a 0.0043 deterioration. This sample cannot resolve an operationally relevant gain; it does not prove no effect. I then quantified a familiar measurement issue: frame-wise nearest-defender identity changed in 36% of one-second windows. A 0.3-metre hysteresis rule reduced switching to 30.4%, but on 1,209 identical shots, none of three defender policies produced detectably better held-out loss. Possession-level probability differences are therefore definition sensitivity, not causal impact. The contribution is not another pressure score. It is an importable audit for event alignment, defender definitions, position uncertainty, and possession replay—making tracking assumptions explicit and testable.

## Submission assets

- `reports/final_report.md`
- `reports/figures/final_summary.png`
- `reports/media/top_disagreement.gif`
- `reports/media/defender_policy_case_1.gif` through `defender_policy_case_3.gif`
- `reports/figures/negative_control.png`
- `reports/tables/`
- reproducible source under `src/` and `scripts/`
