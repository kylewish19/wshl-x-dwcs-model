# DWCS Week 9 — Pre-Fight Winner-Model Development Summary

**Date:** 2026-10-06  
**Timing:** Completed before the scheduled 7:00 p.m. ET Week 9 event start.  
**Integrity:** The original `official_picks/week9_locked_card.csv` and `official_picks/week9_preodds_model_lock.csv` remain unchanged.

## Why this work was done

A Week 9 audit showed that DWCS-W-v0.6 can become overconfident when recent record / win streak / Elo / opponent-quality signals are strong but direct matchup-skill information is thin. The specific trigger was Alexander Chavez being scored at 96.28% over Roque Conceicao even though the production winner input did not contain enough direct striking, grappling, durability, cardio, and defensive evidence to justify treating 96% as literal fight certainty.

## Development candidates tested

### DWCS-W-v0.7 modular challenger

Real ML modules were trained for:
- resume / current form
- striking
- grappling
- durability / finish resistance
- cardio / decision ability
- physicality / opponent quality

A second-stage chronological stacking model combined the module probabilities.

Historical nested validation on 258 fights:
- modular accuracy: 76.74%
- modular Brier: 0.1501
- modular log loss: 0.4516
- full-feature comparison accuracy: 81.40%
- full-feature Brier: 0.1343
- full-feature log loss: 0.4379

**Decision: NOT PROMOTED.** The modular decomposition is useful diagnostically, but replacing the production winner model would reduce historical predictive quality.

### DWCS-W-v0.8 evidence-calibrated challenger

This candidate anchored on the full winner signal and learned whether modular disagreement should attenuate extreme probabilities.

Historical nested validation on 186 fights:
- v0.8 accuracy: 77.96%
- Brier: 0.1583
- log loss: 0.5008
- full-anchor accuracy: 79.57%
- full-anchor Brier: 0.1436
- full-anchor log loss: 0.4758

**Decision: NOT PROMOTED.** It failed all predeclared promotion checks.

### DWCS-W-CG-v0.1 confidence gate

A separate learned model attempted to predict whether the production winner side itself was trustworthy based on raw confidence, module support, module dispersion, and full-vs-module disagreement.

Historical nested validation on 186 fights:
- confidence-gate Brier: 0.1509
- confidence-gate log loss: 0.4823
- raw-anchor Brier: 0.1436
- raw-anchor log loss: 0.4758

**Decision: NOT PROMOTED.** The learned confidence model did not improve calibration.

## Active correction: DWCS-W-SKILL-AUDIT-v0.1

Because none of the new trained challengers beat production, **DWCS-W-v0.6 remains the production side model**.

A new prospective audit / betting-confidence policy is active beginning with the Week 9 pre-fight rerun:

1. Raw W-v0.6 probabilities remain unchanged for grading.
2. Resume/form is now explicitly only one component of the matchup audit.
3. Six separately trained matchup modules are displayed for every fight.
4. Direct skill-stat coverage and tape evidence quality are explicitly reported.
5. Low direct skill coverage can cap or downgrade betting confidence even when W-v0.6 is extreme.
6. Module disagreement can turn a side into PASS / lower-confidence LEAN.
7. No manual replacement probability is invented.
8. Prospectively frozen tape concepts continue accumulating labels so they can eventually receive real trained weights instead of subjective manual weights.

## Week 9 skill-audited rerun

| Fight | Production side | Raw p | Skill-audit status | Audited confidence | Module support |
|---|---|---:|---|---|---:|
| Alivia Bierley vs Summer Onley | Alivia Bierley | 53.35% | PASS | VERY LOW | 3/6 |
| Salhahuddin Everett vs Ozzy Martin | Ozzy Martin | 52.25% | PASS | VERY LOW | 1/6 |
| Roque Conceicao vs Alexander Chavez | Alexander Chavez | 96.28% | LEAN | MEDIUM | 5/6 |
| Ryuho Miyaguchi vs Mateus Soares | Mateus Soares | 98.76% | LEAN | MEDIUM | 6/6 |
| Nell Ariano vs Preston LaGrange | Nell Ariano | 81.05% | LEAN | MEDIUM-HIGH | 6/6 |

### Chavez audit

The six module probabilities for **Conceicao** were:
- resume/form 4.44% -> Chavez strongly
- striking 41.95% -> Chavez narrowly
- grappling 39.90% -> Chavez
- durability 14.15% -> Chavez
- cardio/decision 2.11% -> Chavez strongly
- physical/opponent-quality 56.91% -> Conceicao

Thus the Chavez side survives the broader matchup decomposition, but the 54.8-point module spread and low direct skill-stat coverage make the raw 96.28% inappropriate to interpret as literal certainty. Official audited status is **Chavez LEAN / MEDIUM confidence**.

### Soares audit

All six modules favor Soares, but module probabilities vary materially and direct skill-stat coverage is low. The raw 98.76% is retained for grading but the betting audit is **Soares LEAN / MEDIUM confidence**.

### Ariano audit

All six modules favor Ariano and dispersion is much smaller than Chavez/Soares. This is the strongest audited Week 9 winner side: **Ariano LEAN / MEDIUM-HIGH confidence**.

## Method / duration / round models

No changes were made to DWCS-M-v0.4, DWCS-D-v0.4, or DWCS-R-v0.4 in this audit. Their pre-odds probabilities remain frozen. A fresh 10,000-run Monte Carlo was resampled for each fight and saved in `reports/week9_skill_audited_rerun.json`.

## Files

- `src/build_winner_v0_7_modular.py`
- `reports/winner_v0_7_modular_validation.json`
- `diagnostics/week9_v0_7_modular_rerun.csv`
- `src/build_winner_v0_8_calibrated.py`
- `reports/winner_v0_8_evidence_calibrated_validation.json`
- `src/build_winner_confidence_gate_v0_1.py`
- `reports/winner_confidence_gate_v0_1_validation.json`
- `src/run_week9_skill_audited_rerun.py`
- `diagnostics/week9_skill_audited_rerun.csv`
- `reports/week9_skill_audited_rerun.json`
- `models/model_registry.json`

## Forward rule

Do not claim that all cumulative tape lessons are trained winner-model inputs until they actually have a prospective labeled training history. Continue collecting those fields every DWCS week. The production winner probability and the matchup-skill/betting-confidence audit must be shown separately until a skill-enriched challenger beats W-v0.6 on chronological validation.
