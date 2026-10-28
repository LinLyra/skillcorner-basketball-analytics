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

All models are regularized logistic regressions with preprocessing fit only on training games. The initial regularization parameter is fixed rather than tuned against held-out games. A revised model-class sensitivity analysis nests regularization tuning inside LOGO and adds a small monotonic histogram-gradient tree; it is labelled as a sensitivity analysis rather than part of the original fixed sequence.

## Validation

The primary protocol is leave-one-game-out validation. Metrics are log loss, Brier score, AUC, and calibration-in-the-large. Log loss is primary because the output is a probability and extreme overconfidence should be penalized.

Window lengths are fixed in advance at 0.5, 1.0, and 2.0 seconds. Window comparisons are paired by held-out game. Report the complete per-game distribution; do not present only a pooled score.

With only ten game clusters, the cluster bootstrap is descriptive. Primary revised inference supplements it with all 2^10 exact sign flips and reports the confidence set obtained by inverting the exact two-sided test. A 0.005 log-loss improvement is used as a retrospective practical reference, not a preregistered equivalence margin or a formal minimum detectable effect.

## Robustness checks

Items 1 and 3–7 were in the original protocol. Items 2, 8, and 9 were added in response to methodological review and are reported as revised sensitivity analyses.

1. fixed release defender versus frame-wise nearest defender;
2. a release-anchored 0.3 m hysteresis policy that suppresses near-tie identity changes;
3. all eligible windows versus high detected-rate/low-error windows;
4. raw positions versus smoothed trajectories;
5. inclusion versus exclusion of blocked shots;
6. 5 cm alignment threshold sensitivity;
7. leave-one-game-out coefficient stability and case-level model disagreement;
8. cross-fitted conditional-residual permutation rather than unconditional temporal-vector shuffling;
9. 200 `predError` Monte Carlo reconstructions under the documented isotropic-Gaussian sensitivity assumption.

## Interpretation guardrail

Predictive improvement is not causal evidence that a closeout caused a miss. The study measures incremental predictive information and reliability under the available event/tracking definitions.

Likewise, larger fixed-versus-nearest prediction differences in switching possessions are descriptive. Switching is not randomized and is confounded by crowding, near ties, and defensive complexity. Policy accuracy must be compared on an identical held-out cohort; possession-level differences alone do not establish that a policy is better or that switching caused an outcome.
