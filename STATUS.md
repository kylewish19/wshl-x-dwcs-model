# Current DWCS Model Status

Last updated: 2026-09-16

## Training ledger

- Weeks 2-6 recoverable rows: 25
- Winner record: 16-9
- Week 6 prospective winner record: 4-1

## Active components

### DWCS-CAL-v0.2
- slope: 0.9147857463
- intercept: -0.0205842055
- Brier: 0.2307256699
- log loss: 0.6512034249

### DWCS-M-PRIOR-v0.2
- KO/TKO: 43.5294%
- Submission: 18.6928%
- Decision: 37.7778%

### DWCS-D-1.5-v0.2
- Under 1.5: 41.9737%
- Over 1.5: 58.0263%

## Not yet promoted

### DWCS-W-v0.3
The real champion/challenger training engine exists, but the full historical fighter-level pre-fight matrix is still incomplete. Do not label this model fully trained until walk-forward results exist.

### DWCS-R-v0.1
Exact-round fighter-level model is not promoted yet.

## Week 6 feature changes queued for fighter-level training

- recovery_after_hurt_diff
- scramble_conversion_diff
- counter_grappling_transition_diff
- multi_route_finishing_diff
- early_finish_hazard_diff
- round-dependent cardio treatment

## Next milestone

Build/import the historical DWCS fighter-level matrix, run event-by-event walk-forward validation, export coefficients, and compare the logistic champion against the gradient-boosting challenger before Week 7.
