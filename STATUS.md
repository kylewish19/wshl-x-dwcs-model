# Current DWCS Model Status

Last updated: 2026-09-23

## Production winner model

### DWCS-W-v0.5 — ACTIVE CURRENT-SEASON RETRAIN

- Parent architecture: DWCS-W-v0.4 promoted logistic champion
- Historical DWCS training bouts: 416
- New prospectively locked Week 7 labels appended: 5
- Total fitted rows: 421
- Artifact: `models/dwcs_w_v0_5_week7_logistic.joblib`
- Week 7 coefficient delta: `reports/winner_v0_5_week7_coefficient_delta.csv`

Week 7 changed fitted weights rather than manually editing them. The largest movements were increased global Elo/opponent-quality influence and reduced weight on raw win streak/career volume and prior-DWCS takedown-attempt volume.

DWCS-W-v0.5 inherits the historical validation of v0.4. Week 7 is training data for Week 8 and is not reused as holdout validation.

## Production method model

### DWCS-M-v0.3 — PROMOTED FOR WEEK 8+

Architecture:
- P(winner) × P(method | candidate winner)
- candidate-centered offense
- opponent method-specific vulnerability
- explicit KO access, SUB access, decision route and finish-conversion interactions
- career-route plausibility and model-disagreement gates remain mandatory

Historical future-event walk-forward, 319 held-out DWCS bouts:
- exact winner+method accuracy: 57.99%
- log loss: 1.1937
- multiclass Brier: 0.5569

Previous DWCS-M-v0.2:
- exact winner+method accuracy: 43.57%
- log loss: 1.5068
- multiclass Brier: 0.6886

The v0.3 architecture therefore improves all three primary historical winner+method metrics.

Artifact:
- `models/dwcs_m_v0_3_candidate_method_logistic.joblib`

## Duration / O-U model

### DWCS-D-v0.3 — SHADOW

Week 7 main O/U threshold directions graded 12-3, but that single card is diagnostic only.

The architecture was rebuilt as a coherent survival/hazard chain across:
- 2.5 minutes
- 5 minutes
- 7.5 minutes
- 10 minutes
- 12.5 minutes
- 15 minutes

This guarantees nested probabilities across U/O thresholds, round starts and goes-distance instead of fitting contradictory independent markets.

Historical probability quality still does not beat the simple chronological baseline consistently, so DWCS-D-v0.3 remains shadow.

## Round model

### DWCS-R-v0.3 — SHADOW

R1/R2/R3/DEC probabilities are now derived from the same hazard curve as DWCS-D rather than an independent contradictory model.

Historical walk-forward:
- accuracy: 39.69%
- log loss: 1.4323
- multiclass Brier: 0.7494

Probability quality improved versus the independent v0.2 round model, but still trails the chronological class-frequency baseline. No promotion.

## Week 7 official grade

Official locked Week 7 card:
- Winner: 3-2
- Forced winner+method forecasts: 0-5
- Exact round/decision bucket: 2-3
- O/U 0.5/1.5/2.5 directional calls: 12-3
- Goes distance: 2-3
- Round 2/3 starts: 6-4
- All duration binary shadow markets: 20-10

Week 7 is now stored as labeled training data for Week 8.

## Week 7 lessons encoded

Starting with Week 8 pre-fight capture:
- submission_access_quality_diff
- back_exposure_created_diff
- back_exposure_allowed_diff
- durability_after_clean_damage_diff
- failed_finish_cardio_cost_diff
- late_round_momentum_reversal_diff
- size_physicality_edge_diff
- decision_resilience_diff

These are prospective-only. They are not retroactively fabricated for Week 7 or older fights.

Core method rule:
- weapon ≠ access
- access ≠ conversion
- historical finish rate alone cannot overrule opponent-specific durability, recovery or route vulnerability

## Temporal-integrity correction

Week 6 model predictions were generated after the September 15 event. Week 6 is therefore a retrospective/backtest diagnostic, not a prospective lock and not clean prospective validation.

The calibration fit that included Week 6 post-event probabilities is marked retrospective/development-only. The last clean recoverable calibration fallback is DWCS-CAL-v0.1 through Week 5.

## Forward weekly cycle

For every new DWCS card:

1. Freeze current fighter records, regional history and tape features before the event.
2. Run DWCS-W, DWCS-M, DWCS-D and DWCS-R.
3. Run fresh 10,000-fight simulations.
4. Lock the card before odds are allowed to influence bet selection.
5. Grade winner, winner+method, totals and round outputs separately.
6. Append verified outcomes.
7. Refit all four systems.
8. Save coefficient/model deltas and only promote challengers that earn it on chronological validation.
