# Fighter-Level Feature Contract

One row represents one historical DWCS bout. Every feature must be frozen immediately **before** that bout.

## Core identifiers

- event_date
- event_id
- fighter_a
- fighter_b
- winner_a

## Pre-fight numeric differences

- age_diff
- height_diff
- reach_diff
- pro_fights_diff
- wins_diff
- losses_diff
- career_win_pct_diff
- recent3_win_pct_diff
- recent5_win_pct_diff
- career_finish_pct_diff
- ko_win_pct_diff
- sub_win_pct_diff
- finish_loss_pct_diff
- opponent_win_pct_diff
- quality_win_count_diff
- days_since_last_fight_diff
- regional_strength_diff

## Prior-DWCS statistics when available

- prior_dwcs_bouts_diff
- dwcs_sig_str_landed_pm_diff
- dwcs_sig_str_absorbed_pm_diff
- dwcs_strike_accuracy_diff
- dwcs_strike_defense_diff
- dwcs_td_landed_per15_diff
- dwcs_td_accuracy_diff
- dwcs_td_defense_diff
- dwcs_control_share_diff
- dwcs_knockdowns_per15_diff
- dwcs_sub_attempts_per15_diff
- prior_dwcs_stats_available_diff

## Tape / research features

All tape features must use a fixed prospective scoring rubric.

- initial_takedown_access_diff
- sustained_takedown_access_diff
- finish_access_diff
- repeat_takedown_access_diff
- top_control_retention_diff
- getup_countergrapple_diff
- front_headlock_exposure_diff
- offensive_wrestling_counter_exposure_diff
- pressure_cardio_diff
- recovery_durability_diff
- recovery_after_hurt_diff
- scramble_conversion_diff
- counter_grappling_transition_diff
- late_power_diff
- multi_route_finishing_diff
- early_finish_hazard_diff
- missing_defensive_evidence_diff

## Week 6 rule changes

- Cardio is round-sensitive rather than a single flat winner penalty.
- Durability should influence method/duration more than winner probability.
- Ultra-early finish hazard must be matchup-specific, not globally inflated because one card had fast finishes.
- Unknown defensive evidence is uncertainty, not assumed average defense.

## Week 7 rule changes

These features begin with Week 8 pre-fight capture. They are **not** retroactively assigned to Week 7 or older fights after seeing results.

- submission_access_quality_diff
- back_exposure_created_diff
- back_exposure_allowed_diff
- durability_after_clean_damage_diff
- failed_finish_cardio_cost_diff
- late_round_momentum_reversal_diff
- size_physicality_edge_diff
- decision_resilience_diff

Method-specific rules:
- Separate weapon from access and access from conversion.
- Candidate method offense and opponent method-specific vulnerability must be represented separately; A-minus-B differences alone are not sufficient for DWCS-M.
- Career route evidence is a plausibility check, not a hard manual method override.
- High historical finish rate does not automatically beat decision when opponent durability/recovery evidence is strong.
- Back exposure and strike-to-grapple transitions can raise submission probability even for fighters whose historical wins skew KO/TKO.

Duration/round rules:
- Independent O/U thresholds may not be promoted if they violate probability nesting.
- The preferred challenger is a coherent survival/hazard layer across 2.5, 5, 7.5, 10, 12.5 and 15 minutes.
- Round probabilities should be derived from the same survival curve whenever possible so R1/R2/R3/DEC and O/U markets cannot contradict one another.

## Anti-leakage

- No current-fight odds.
- No current-fight result, method, or round.
- No statistics accumulated after event_date.
- Week N results may train Week N+1 only.
- A feature invented after a card may be captured prospectively starting the next card, but may not be retrospectively fabricated for earlier bouts.
