# DWCS Week 7 Pre-Odds Model Audit

Event date: 2026-09-22

## Status

This is the audited pre-odds lock. The raw four-model output remains preserved for diagnostics, but it is not the final betting gate.

No sportsbook odds were used.

## Winner audit

The production DWCS-W-v0.4 logistic output was compared with:
- DWCS-W-v0.3 historical logistic fallback
- DWCS-W-v0.4 histogram-boosting challenger

The official side is unchanged from the production model, but confidence is reduced when the three winner models disagree or when the fallback/challenger sit near 50/50. The three-model median is a diagnostic only; it is not claimed as a separately validated probability model.

## Method audit

DWCS-M-v0.2 exposed an out-of-distribution problem on Theo Haig: the conditional model assigned nearly all of Haig's win probability to KO/TKO even though his professional wins are 7 submissions and 0 knockouts. Therefore the conditional method model cannot promote a route by itself.

Pre-odds method gate:
1. Winner side must survive the winner-model audit.
2. Conditional method and direct six-class model are compared after conditioning the direct model on the selected winner.
3. Career route must be plausible.
4. A method is bet-qualified only when winner confidence and route probability are both strong.
5. Otherwise retain a forecast but mark SIDE ONLY / METHOD LEAN.

Week 7 result: Emilio Quissua KO/TKO is the only method route that clears the current pre-odds qualification gate. It still requires sportsbook price review before becoming a wager.

## Duration and round audit

DWCS-D-v0.2 and DWCS-R-v0.2 remain shadow models. Week 7 produced internal cross-market inconsistencies, including cases where independently fitted survival thresholds violate logical monotonicity and cases where the round/distance model strongly conflicts with the method model.

Therefore no Week 7 total or exact-round output is promoted as an official bet before odds. Shadow leans are retained for learning and later grading.

Next architecture change: replace independently fitted duration thresholds with a coherent survival/hazard layer so probabilities for U/O thresholds, round starts, distance and finish-round buckets obey nesting constraints.

## Locked sides

- Theo Haig over Norbert Novenyi Jr. — LOW
- Alvi Dasuyev over Jaden Ortega — LOW
- Marcos Degli over Paris Moran — HIGH
- Piero Guaylupo over Callum Connor — LOW
- Emilio Quissua over Damian Piwowarczyk — MEDIUM-HIGH

## Method forecasts

- Haig SUB — side only/no method bet (conditional model failed plausibility check)
- Dasuyev SUB — side only/no method bet
- Degli KO/TKO — method lean
- Guaylupo KO/TKO — method lean
- Quissua KO/TKO — method qualified; price pending

This audit was completed before sportsbook odds were introduced.
