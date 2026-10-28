# NBA analytics portfolio brief

## Project summary

Built an end-to-end, reproducible analysis of 25 Hz professional basketball tracking data to test whether pre-release defensive movement improves shot-quality estimation beyond a static snapshot.

## What this demonstrates

- normalized 10 games of nested tracking and event data;
- validated event-to-tracking joins and identified silent distance mismatches;
- engineered interpretable temporal and multiplayer spatial features;
- designed leakage-resistant leave-one-game-out evaluation;
- evaluated probability calibration, uncertainty, negative controls, and subgroup stability;
- compared open models with a vendor score as an external reference with unknown training overlap;
- translated model disagreements into possession-level visual explanations;
- packaged an importable game-audit API and CLI with pinned data versions and automated tests.

## Résumé bullet

Built a reproducible 25 Hz professional-basketball tracking pipeline covering 1,459 shots; audited event alignment and defender identity, propagated vendor position-error bounds, and evaluated temporal and multiplayer defensive features with leave-one-game-out validation, exact sign flips, conditional-residual permutation, nested tuning, and an external Shot Quality reference.

## Interview discussion point

The strongest result was not a higher headline AUC. Exact small-cluster inference showed that ten public games cannot resolve the small temporal point estimate and still cannot rule out a practically relevant gain. A separate defender-policy audit quantified identity switching, showed that a 0.3 m hysteresis rule reduces it, and found no detectable held-out accuracy advantage for any policy. That distinction turned feature engineering into a transparent measurement-reliability study without making causal or no-effect claims.
