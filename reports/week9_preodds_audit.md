# DWCS Week 9 Pre-Odds Audit

Event date: 2026-10-06

## Integrity

All five submitted Week 9 matchups were verified against current UFC/Sherdog listings. All ten fighters made weight. No sportsbook odds were used.

Production stack:
- DWCS-W-v0.6 winner
- DWCS-M-v0.4 hierarchical finish-route
- DWCS-D-v0.4 pair-symmetric coherent hazard
- DWCS-R-v0.4 hazard-derived round bucket
- 10,000 Monte Carlo simulations per fight

The current winner coverage gate is enforced. If both regional histories are not established, the official side falls back to DWCS-W-v0.3. On fallback fights the same fallback winner probability is now also used for joint winner+method probabilities and Monte Carlo, preventing mixed winner components.

## Winner audit

### Alivia Bierley vs Summer Onley
- Official gate result: V0.3 fallback -> Bierley 53.35%.
- W-v0.6 raw: Bierley 51.25%.
- v0.6 boosting shadow: Onley 72.88%.
- Major architecture disagreement; VERY LOW / PASS.

### Salhahuddin Everett vs Ozzy Martin
- Official gate result: V0.3 fallback -> Martin 52.25%.
- W-v0.6 raw: Everett 56.42%.
- v0.6 boosting shadow: Everett 72.77%.
- Official fallback conflicts with both current models; VERY LOW / PASS.

### Roque Conceicao Moreira Junior vs Alexander Chavez
- W-v0.6: Chavez 96.28%.
- boosting shadow: Chavez 96.91%.
- v0.3 fallback: Chavez 50.20%.
- All models are on Chavez, but the old fallback is essentially a coin flip; audited MEDIUM-HIGH rather than treating 96% as literal certainty.

### Ryuho Miyaguchi vs Mateus Soares
- W-v0.6: Soares 98.76%.
- boosting shadow: Soares 99.27%.
- v0.3 fallback: Soares 65.09%.
- All three architectures agree. This is the strongest winner side on the card; audited HIGH, not a guaranteed outcome.

### Nell Ariano vs Preston LaGrange
- W-v0.6: Ariano 81.05%.
- boosting shadow: Ariano 87.41%.
- v0.3 fallback: LaGrange 57.46%.
- Primary current models agree strongly but the older fallback disagrees; audited MEDIUM.

## Method audit

No method officially clears the full pre-odds qualification gate.

- Bierley DEC: forecast only. Winner side is a fallback near coin flip.
- Martin DEC: forecast only and route plausibility is weak; Martin has no pro decision wins.
- Chavez DEC: model forecast, but cross-model probability spread prevents qualification.
- Soares KO/TKO: strongest method lean. Conditional on a Soares win, M-v0.4 assigns 56.18% KO/TKO and the joint exact probability is 55.49%. Not qualified pre-odds because the winner-model probability spread narrowly exceeds the method gate.
- Ariano KO/TKO: M-v0.4 is extremely strong conditionally (91.51%; 74.17% joint), but the v0.3 winner fallback is on LaGrange, so it remains a lean rather than an official method qualification.

## Duration / O-U production outputs

DWCS-D-v0.4 is now production, but price still determines whether a probability is a bet.

- Bierley-Onley: O1.5 61.04%; distance 50.94%; R2 starts 63.58%.
- Everett-Martin: O1.5 58.83%; R2 starts 63.22%; distance 45.85%.
- Conceicao-Chavez: O1.5 71.62%; R2 starts 74.99%; R3 starts 60.46%; distance 50.17%.
- Miyaguchi-Soares: U2.5 82.05%; No Distance 84.46%; R1 50.07%.
- Ariano-LaGrange: U2.5 70.97%; No Distance 79.19%; R1 34.50%.

## Prospective lesson-feature integrity

All accumulated lessons through Week 8 were carried forward. New Week 8 features (elite amateur pedigree, amateur sample, pre-UFC championship experience, five-round experience, grappling-generated TKO access, slam/ground-damage TKO access, early-finish route strength and post-high-output-round freshness) were frozen before Week 9 results and are stored in `data/current/week9_tape_features.csv`.

These newly introduced lesson features are audit/training fields for Week 9; they are not retrospectively assigned to older bouts and are not manually used as hidden probability bumps.
