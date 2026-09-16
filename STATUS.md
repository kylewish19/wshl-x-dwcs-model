# Current DWCS Model Status

Last updated: 2026-09-16

## Historical fighter-level winner model

### DWCS-W-v0.3 — PROMOTED BASELINE

Training matrix:
- 416 usable DWCS bouts
- 86 events
- 751 fighters
- historical range: 2017-07-11 through 2025-09-24
- DWCS only; no UFC training rows

Event-by-event walk-forward validation:
- 66 future-event folds
- 320 held-out bouts

Champion — L2 logistic regression:
- accuracy: 61.25%
- Brier: 0.2341
- log loss: 0.6828
- AUC: 0.6747

Challenger — histogram gradient boosting:
- accuracy: 60.31%
- Brier: 0.2472
- log loss: 0.7017
- AUC: 0.6413

Decision: keep the logistic champion. It beat the challenger on both probability-quality metrics.

Important limitation: this historical source contains physical attributes and prior-DWCS history/stats but not the complete regional career ledger that most DWCS debutants bring into their first appearance. Therefore DWCS-W-v0.3 is a real ML baseline component, not the finished all-information winner system.

## Current-season learning layers

- Weeks 2-6 recoverable winner ledger: 16-9
- Week 6 prospective winners: 4-1
- calibration layer trained weekly
- method population prior trained weekly
- O/U 1.5 population prior trained weekly

## Week 6 lessons in feature contract

- recovery_after_hurt_diff
- scramble_conversion_diff
- counter_grappling_transition_diff
- multi_route_finishing_diff
- early_finish_hazard_diff
- round-dependent cardio treatment

These tape-derived variables are not retroactively fabricated for historical fights.

## Next upgrade

1. Add pre-fight regional career record / opponent-quality features.
2. Create and save the actual pre-fight feature row for every Week 7+ matchup.
3. After each card, append labels and retrain DWCS-W so the fitted coefficients actually change week to week.
4. Keep a coefficient-delta report for every retrain.
