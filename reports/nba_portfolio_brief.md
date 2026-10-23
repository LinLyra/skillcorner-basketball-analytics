# NBA analytics portfolio brief

## Project summary

Built an end-to-end, reproducible analysis of 25 Hz professional basketball tracking data to test whether pre-release defensive movement improves shot-quality estimation beyond a static snapshot.

## What this demonstrates

- normalized 10 games of nested tracking and event data;
- validated event-to-tracking joins and identified silent distance mismatches;
- engineered interpretable temporal and multiplayer spatial features;
- designed leakage-resistant leave-one-game-out evaluation;
- evaluated probability calibration, uncertainty, negative controls, and subgroup stability;
- benchmarked open models against a production-grade vendor metric;
- translated model disagreements into possession-level visual explanations;
- maintained pinned data versions and automated tests.

## Résumé bullet

Built a reproducible 25 Hz professional-basketball tracking pipeline covering 1,459 shots; audited event alignment and defender identity, propagated vendor position-error bounds, and evaluated temporal and multiplayer defensive features with leave-one-game-out validation, exact sign flips, conditional-residual permutation, nested tuning, and an external Shot Quality reference.

## Interview discussion point

The strongest finding was not a higher headline AUC. It was discovering that aligned temporal features contain real shot-specific information while still failing to produce a stable cross-game gain. That distinction changed the project from feature engineering into a reliability study and prevented an unsupported performance claim.
