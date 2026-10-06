# Current DWCS Model Status

Last updated: 2026-10-06

## Current production stack

- Winner: **DWCS-W-v0.6**
- Winner + method: **DWCS-M-v0.4**
- Duration / O-U / distance / round starts: **DWCS-D-v0.4**
- Exact round bucket: **DWCS-R-v0.4**
- Clean calibration fallback: **DWCS-CAL-v0.1-clean**

All current models preserve the cumulative lessons from prior DWCS weeks. New lesson features are only populated prospectively; they are never invented retrospectively after outcomes are known.

## Winner model

### DWCS-W-v0.6 — ACTIVE

Training state:
- 416 historical DWCS bouts
- 5 prospectively locked Week 7 rows
- 4 completed Week 8 rows
- 425 total fitted rows
- Bulaid-Visconde excluded because the fight was cancelled after weigh-ins

Artifact:
- `models/dwcs_w_v0_6_week8_logistic.joblib`

Week 7 + Week 8 prospective diagnostic sample, 9 completed fights:
- production logistic accuracy: 66.67%
- production Brier: 0.1832
- production log loss: 0.4972
- boosting challenger accuracy: 66.67%
- boosting Brier: 0.2362
- boosting log loss: 0.7261
- simple 50/50 ensemble accuracy: 66.67%
- ensemble Brier: 0.1847
- ensemble log loss: 0.5461

Decision: keep the logistic model as production. Boosting and the simple ensemble remain shadow because neither improved prospective probability quality.

Week 8 refitting increased global Elo influence while reducing raw career-volume influence. These were learned coefficient changes, not manual edits.

## Method model

### DWCS-M-v0.4 — PROMOTED

Architecture:
- P(winner)
- × P(finish vs decision | candidate winner)
- × P(KO/TKO vs submission | finish, candidate winner)

The method problem is now split into two questions instead of forcing one three-way classifier to simultaneously decide finish/decision and finish family.

Historical future-event walk-forward, 319 held-out DWCS bouts:
- exact winner+method accuracy: **57.99%**
- log loss: **1.1879**
- multiclass Brier: **0.5525**

Previous DWCS-M-v0.3:
- exact winner+method accuracy: 57.99%
- log loss: 1.1937
- multiclass Brier: 0.5569

Promotion decision: v0.4 improved both probability-quality metrics without reducing exact winner+method accuracy, satisfying the predeclared promotion rule.

Artifacts:
- `models/dwcs_m_v0_4_finish_gate_logistic.joblib`
- `models/dwcs_m_v0_4_finish_route_logistic.joblib`

Fallback:
- `models/dwcs_m_v0_3_week8_refit.joblib`

Important Week 8 method lessons now represented in the architecture/feature contract:
- opponent submission vulnerability cannot overpower the winner's repeated KO/TKO route by itself
- historical standing-KO count is not the only KO/TKO path; slams, wrestling and ground damage matter
- finish-vs-decision is separated from KO-vs-SUB routing
- contextual official results remain labels but carry audit flags

## Duration / O-U model

### DWCS-D-v0.4 — PROMOTED

This is the biggest structural Week 8 improvement.

The previous timing models relied heavily on signed A-minus-B differences. That is inappropriate for fight duration because swapping which fighter is listed as A or B should not change how long the fight is expected to last.

DWCS-D-v0.4 uses **30 order-invariant pair features**, including:
- pair finish-rate mean/max/min/product
- recent finish pressure
- R1/early-finish pressure
- pair finish-loss vulnerability
- minimum durability
- KO/SUB route pressure
- pair average duration
- total experience
- absolute Elo/experience gaps
- finish-pressure × vulnerability interactions

It still uses one coherent survival chain at:
- 2.5 minutes
- 5 minutes
- 7.5 minutes
- 10 minutes
- 12.5 minutes
- 15 minutes

So all O/U, round-start and distance probabilities remain mathematically consistent.

Historical future-event walk-forward, 319 bouts:

| Market | Accuracy | Brier | Log loss | AUC |
|---|---:|---:|---:|---:|
| U0.5 | 85.58% | 0.1120 | 0.3715 | 0.741 |
| U1.5 | 68.97% | 0.2005 | 0.5844 | 0.728 |
| U2.5 | 65.20% | 0.2118 | 0.6056 | 0.720 |
| Goes Distance | 66.14% | 0.2005 | 0.5781 | 0.740 |
| Round 2 Starts | 72.41% | 0.1881 | 0.5574 | 0.730 |
| Round 3 Starts | 65.20% | 0.2109 | 0.6050 | 0.724 |

**All six timing markets beat the chronological naive baseline on both Brier score and log loss.**

Artifacts:
- `models/dwcs_d_pair_hazard_*_v0_4.joblib`

This earns production promotion. Week 9+ will be the start of its clean prospective record.

## Round model

### DWCS-R-v0.4 — PROMOTED

Exact R1/R2/R3/DEC is derived from the same pair-symmetric survival curve as DWCS-D-v0.4.

Historical future-event walk-forward:
- exact round/decision accuracy: **51.72%**
- log loss: **1.1563**
- multiclass Brier: **0.6203**

Chronological class-frequency baseline:
- accuracy: 39.18%
- log loss: 1.2616
- multiclass Brier: 0.6911

DWCS-R-v0.4 beat the baseline on accuracy and both probability-quality metrics, so it is now production eligible and promoted.

## Week 8 official grade

Bulaid-Visconde was cancelled after weigh-ins and excluded from all grades and training labels.

Completed-card grades:
- production winner: **3-1 (75%)**
- official side leans: **2-0**
- boosting challenger: **3-1**
- forced exact winner+method: **1-3**
- official method leans: **0-1**
- main shadow total leans: **1-3**
- all O/U thresholds: **5-7**
- goes distance: **2-2**
- round-start markets: **2-6**
- all duration binary markets: **9-15**
- exact round/decision bucket: **1-3**

The poor Week 8 D/R prospective grade is preserved. Promotion of v0.4 is based on a newly designed, order-invariant architecture and historical chronological validation, not on retroactively changing Week 8 predictions.

## Prospective features added after Week 8

Starting with the next card:
- elite_amateur_pedigree_diff
- amateur_fight_sample_diff
- pre_ufc_championship_experience_diff
- five_round_experience_diff
- grappling_generated_tko_access_diff
- slam_ground_damage_tko_access_diff
- early_finish_route_strength_diff
- post_high_output_round_freshness_diff

These join, rather than replace, all earlier cumulative lesson features from Weeks 2-7.

## Temporal-integrity rules

- Week 6 predictions remain retrospective diagnostic only.
- Week 7 was the first clean prospective lock in the current GitHub workflow.
- Week 8 locked probabilities remain unchanged after grading.
- Cancelled fights are not counted as wins/losses and do not become training labels.
- A new feature discovered after Week N begins with Week N+1 prospectively and is not fabricated for older fights.
- Current odds are never model inputs before the pre-odds lock.

## Forward weekly cycle

1. Freeze current records, regional history, amateur/championship context and tape features.
2. Run W / M / D / R.
3. Run fresh 10,000 simulations per matchup.
4. Lock the pre-odds card.
5. Add sportsbook prices only after the lock.
6. Grade winner, exact winner+method, O/U, distance, round starts and exact round separately.
7. Append only completed verified outcomes.
8. Refit all four systems.
9. Save coefficient/model changes and promote challengers only when chronological validation earns it.
