# Analytics Cup submission package

## Title

**Beyond the Snapshot: When Does Temporal Defensive Context Improve Basketball Shot Quality?**

## One-sentence contribution

A reproducible, uncertainty-aware evaluation shows that correctly aligned pre-release motion contains shot-specific signal, but simple temporal and multiplayer geometry features do not deliver stable cross-game gains beyond release distance—and do not improve SkillCorner’s official Shot Quality.

## Short abstract

Tracking data invites increasingly complex descriptions of defensive pressure, but more motion features do not necessarily generalize. Using ten SkillCorner professional basketball games at 25 Hz, this project compares location/context, release-distance, temporal-defense, and multiplayer-geometry models under leave-one-game-out validation. It audits event/tracking alignment, defender identity, extrapolated positions, smoothing, basketball subgroups, and within-game permutation controls. A one-second temporal model slightly improves mean log loss over a static model, yet the paired game-level interval crosses zero. Correctly aligned temporal features outperform 97.5% of permutation controls, suggesting real signal without stable operational gain. Defender identity switches in 36% of frame-wise-nearest trajectories, while a larger multiplayer model generalizes worse. SkillCorner’s official Shot Quality remains substantially stronger, and adding the simple temporal block does not improve it. The work contributes an open reliability protocol, negative results that constrain future model design, and possession-level visual explanations.

## 60-second presentation script

> Shot quality is often judged from a single frame: where was the shooter, and how close was the defender? With tracking data, we can look backward—but does that actually help? I tested 1,459 shots from ten professional games using strict leave-one-game-out validation. A one-second defensive history performed slightly better than release distance alone, but the improvement was small and inconsistent across games. Importantly, correctly aligned motion beat 97.5% of within-game permutation controls, so the signal is not random. Yet changing defender definitions, filtering tracking quality, smoothing trajectories, and adding multiplayer geometry did not produce a stable gain. In fact, the nearest defender changed identity during 36% of one-second windows. SkillCorner’s official Shot Quality remained much stronger, and my temporal summaries did not improve it. The result is a practical warning: more tracking features are not automatically better. Reliable basketball analytics needs explicit defender definitions, alignment checks, uncertainty, and strong external benchmarks.

## Submission assets

- `reports/final_report.md`
- `reports/figures/final_summary.png`
- `reports/media/top_disagreement.gif`
- `reports/figures/negative_control.png`
- `reports/tables/`
- reproducible source under `src/` and `scripts/`
