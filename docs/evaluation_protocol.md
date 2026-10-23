# Evaluation protocol

## Confirmatory question

Does pre-release temporal defensive context improve out-of-game shot-outcome probability estimates beyond location, shot context, and release-time defender distance?

## Unit of analysis

One row is one shot. Tracking frames are repeated observations used to derive features and are never treated as independent outcome samples.

## Eligibility

- tracking and events align at the release frame;
- shooter and official release defender are present;
- the recomputed and official release distances differ by no more than 0.05 m;
- the shot is not marked as fouled.

Blocked shots remain in the primary analysis. Their exclusion is a planned sensitivity analysis.

## Fixed model sequence

- M0: shot distance, shot clock, three-point flag, catch-and-shoot flag, prior dribbles, region, and shot type;
- M1: M0 plus defender distance at release;
- M2: M1 plus recent closing speed, separation gain, pressure duration, and window distance summaries.

All models are regularized logistic regressions with preprocessing fit only on training games. The initial regularization parameter is fixed rather than tuned against held-out games.

## Validation

The primary protocol is leave-one-game-out validation. Metrics are log loss, Brier score, AUC, and calibration-in-the-large. Log loss is primary because the output is a probability and extreme overconfidence should be penalized.

Window lengths are fixed in advance at 0.5, 1.0, and 2.0 seconds. Window comparisons are paired by held-out game. Report the complete per-game distribution; do not present only a pooled score.

## Planned robustness checks

1. fixed release defender versus frame-wise nearest defender;
2. all eligible windows versus high detected-rate/low-error windows;
3. raw positions versus smoothed trajectories;
4. inclusion versus exclusion of blocked shots;
5. 5 cm alignment threshold sensitivity;
6. leave-one-game-out coefficient stability and case-level model disagreement.

## Interpretation guardrail

Predictive improvement is not causal evidence that a closeout caused a miss. The study measures incremental predictive information and reliability under the available event/tracking definitions.
