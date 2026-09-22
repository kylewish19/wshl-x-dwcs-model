# Current DWCS Model Status

Last updated: 2026-09-16

## Production winner model

### DWCS-W-v0.4 — PROMOTED LOGISTIC CHAMPION

Historical label set:
- 416 usable DWCS bouts
- 86 events
- 751 DWCS fighters
- 320 held-out bouts across 66 future-event walk-forward folds
- DWCS fights only are prediction labels

External pre-fight state:
- MMA Global Database career history
- 125,539 source fights loaded before the historical DWCS cutoff
- record/form/finish mix/opponent quality/Elo/activity features only
- external fights are state inputs, never DWCS prediction labels
- raw external database is not committed to this repository

Conservative anti-leakage audit:
- accuracy: 82.50%
- Brier: 0.1324
- log loss: 0.4298
- AUC: 0.8951
- date-leakage violations: 0
- explicit history-availability feature removed in audit
- missingness indicators disabled in audit
- shuffled regional-feature control: 58.44% accuracy, 0.7153 log loss

### Coverage gate

The regional model is strongest only when both fighters have established pre-fight career histories.

Historical held-out segment results:
- both histories matched: 217 fights, 91.71% accuracy, 0.2895 log loss
- one history matched: 89 fights, 61.80% accuracy, 0.7547 log loss
- neither matched: 14 fights, too small for a reliable standalone conclusion

Production rule:
- use DWCS-W-v0.4 as the primary ML winner component only when both fighters' current career histories are established;
- if one/both histories cannot be established, fall back to DWCS-W-v0.3 plus current tape/research rather than treating missing history as neutral.

### Champion vs challenger

Logistic remains champion because:
- it supports the requested week-to-week coefficient audit;
- it had better accuracy and Brier than the boosting challenger;
- it passed the conservative no-missingness audit.

Histogram gradient boosting remains challenger:
- accuracy: 78.44%
- Brier: 0.1365
- log loss: 0.4193
- AUC: 0.8938

Its log loss was slightly better, so it remains useful as a disagreement check but does not replace the logistic champion.

## Current-season learning

- Weeks 2-6 recoverable winner ledger: 16-9
- Week 6 prospective winner record: 4-1
- calibration layer retrains weekly
- method population prior retrains weekly
- O/U 1.5 population prior retrains weekly
- coefficient-delta reporting is enabled after each retrain

## Week 6 lessons in feature contract

- recovery_after_hurt_diff
- scramble_conversion_diff
- counter_grappling_transition_diff
- multi_route_finishing_diff
- early_finish_hazard_diff
- round-dependent cardio treatment

These tape-derived features are saved prospectively; they are not retroactively fabricated for old fights.

## Current limitation / Week 7 rule

The global career source currently ends 2026-01-31. Every 2026 card therefore requires a fresh current-record/tape research pass before inference. Stale January records must never be presented as current fighter state.

## Next milestone

Build the Week 7 pre-fight feature capture format so the current regional record, opponent-quality state, tape scores, and model probabilities are frozen in GitHub before the card. After Week 7, append the outcomes and produce a real coefficient-delta report.


## Week 7 preflight audit — 2026-09-22

The live pre-odds run exposed two important runtime issues before results were known:

1. **DWCS-M-v0.2 conditional method requires a plausibility gate.** Theo Haig's conditional output incorrectly concentrated on KO/TKO despite a 7-SUB / 0-KO professional win history. The direct six-class model and career route both favored submission. Going forward, no method is promoted from the conditional model alone.
2. **DWCS-D/R remain shadow-only.** Independent duration classifiers produced non-nested survival probabilities on Week 7 and some severe method/distance conflicts. The next duration architecture should be a coherent survival/hazard model.

The audited pre-odds card is:
- `official_picks/week7_preodds_audited_lock.csv`
- audit notes: `reports/week7_preodds_audit.md`

These changes were made before sportsbook odds and before fight outcomes.
