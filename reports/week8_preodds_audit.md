# DWCS Week 8 Pre-Odds Audit

Event date: 2026-09-29

## Card and weigh-in integrity

All five submitted matchups were verified against current UFC/Sherdog listings. All ten fighters made weight.

Record reconciliation follows the same sanctioned-pro convention used by the project. George Staines is stored as 8-0 in Sherdog/UFC sanctioned-pro records even though some sources count two professional exhibitions and display 10-0. Zaurbek Sabanov is stored as 5-0 because Sherdog counts the 2024 UMMA bout as professional while the UFC preview displays 4-0. These source discrepancies are documented rather than silently blended.

## Tape/research evidence classification

- Ilias Bulaid: full-fight source located (UAE Warriors 41) plus current UFC/Sherdog history; direct frame-by-frame playback was not available in this tool run.
- Erick Visconde: current UFC profile/preview, LFA/Sherdog fight history and route evidence; no full-fight playback verified in this run.
- Aieza Ramos Bertolso: multiple full amateur fight sources located plus LFA title history and UFC preview.
- Camila Reynoso: current UFC/Sherdog fight history and matchup preview; no full-fight playback verified.
- Luca Borando: full recent Cage Warriors fight source located plus current UFC/Sherdog history.
- Ian Escuza: current UFC/Sherdog history and matchup preview; no full-fight playback verified.
- Zaurbek Sabanov / Adama Diop: current UFC preview plus complete result histories; no full-fight playback verified.
- Loai Abushaar / George Staines: current UFC preview plus Sherdog/Tapology/FightMatrix history; no full-fight playback verified.

Missing film evidence is treated as uncertainty, not as average or positive defense.

## Winner audit

Raw production output comes from DWCS-W-v0.5. Confidence is reduced when the v0.3 fallback and v0.4 boosting challenger disagree materially.

- Visconde: all three models choose Visconde, but the v0.3 fallback is only narrowly on that side. MEDIUM.
- Reynoso: production and v0.3 narrowly choose Reynoso while boosting strongly chooses Bertolso. LOW / PASS.
- Escuza: production and boosting choose Escuza; v0.3 chooses Borando. LOW-MEDIUM / LEAN.
- Diop: production narrowly chooses Diop while both diagnostics choose Sabanov. VERY LOW / PASS.
- Staines: all three choose Staines, but v0.3 is much less confident. MEDIUM / LEAN.

## Method audit

- Visconde DEC is career-plausible (five decision wins, current five-round LFA title win) and remains a method lean.
- Reynoso DEC is not promoted because the winner side itself is low confidence.
- Escuza SUB is opponent-specific rather than career-frequency driven: Borando's only loss is by rear-naked choke and Escuza owns two submission wins. It remains a lean, not a qualified bet.
- Diop DEC is not promoted because the winner side is a near coin flip with diagnostic disagreement.
- Staines KO/TKO is rejected. The raw model assigns KO/TKO highest conditional probability, but Staines has only one KO/TKO win and five decision wins in the sanctioned 8-0 pro record. This is treated as an out-of-distribution route conflict, so the official card is SIDE ONLY for Staines.

## Duration / round

DWCS-D-v0.3 and DWCS-R-v0.3 remain shadow-only. Their probabilities are coherent by construction now, but they have not earned production promotion historically. Shadow total directions are saved for grading.

No sportsbook odds were used.
