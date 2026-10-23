# Analytics Cup submission package

## Title

**Beyond the Snapshot: When Does Temporal Defensive Context Improve Basketball Shot Quality?**

## One-sentence contribution

A reproducible audit shows that simple pre-release motion summaries do not deliver stable incremental value beyond release distance, and that defender identity choices can materially change possession-level conclusions.

## Short abstract

Tracking data invites increasingly complex descriptions of defensive pressure, but more motion features do not necessarily generalize. Using ten SkillCorner professional basketball games at 25 Hz, this project compares location/context, release-distance, temporal-defense, and multiplayer-geometry models under leave-one-game-out validation. A one-second temporal model changes mean log loss by −0.00125, with a game-cluster interval [−0.00586, 0.00316] and exact sign-flip p = 0.613. A cross-fitted conditional permutation also fails to establish incremental information (15.5% as favorable as observed). The conclusion persists with nested tuning and a monotonic nonlinear tree. Defender identity switches in 36% of frame-wise-nearest trajectories; switched possessions produce over three times the average prediction discrepancy of stable ones. The project contributes an open audit protocol, explicit position-error propagation, honest boundary results, and possession-level visual explanations. SkillCorner Shot Quality is shown only as an external production reference because its training provenance and overlap are unknown.

## 60-second presentation script

> Shot quality is often judged from one frame: where was the shooter, and how close was the defender? I tested whether the previous second adds value, using 1,459 shots from ten professional games and strict leave-one-game-out validation. The temporal model looked slightly better on average, but an exact ten-game test gave p equals 0.61, and a conditional permutation found no reliable information beyond static context. That null result survived tuning and a nonlinear model. The bigger discovery was measurement: the “nearest defender” changed identity in 36% of one-second windows. When it switched, two reasonable defender policies changed predicted make probability by 2.6 percentage points on average—over three times the stable cases. So this is not another black-box pressure score. It is an open audit showing when tracking metrics change the basketball question they claim to answer, with event-alignment checks, position-error propagation, and possession-level replays. More tracking features are not automatically better; their definitions must survive scrutiny.

## Submission assets

- `reports/final_report.md`
- `reports/figures/final_summary.png`
- `reports/media/top_disagreement.gif`
- `reports/media/defender_policy_case_1.gif` through `defender_policy_case_3.gif`
- `reports/figures/negative_control.png`
- `reports/tables/`
- reproducible source under `src/` and `scripts/`
