# DWCS Season 10 Week 7 — Current Record Reconciliation

Research cutoff: 2026-09-16

The five scheduled matchups are saved in `data/current/week7_matchups.csv`.

Current professional records are reconciled against the live Sherdog fighter pages rather than copied from aggregator cards. This matters because other current sites disagree on several records (notably Jaden Ortega, Paris Moran, and Callum Connor).

## Reconciled records

- Norbert Novenyi Jr. 10-1
- Theo Haig 7-1
- Alvi Dasuyev 9-0
- Jaden Ortega 6-0
- Marcos Degli 14-3
- Paris Moran 14-3 with 1 NC
- Piero Guaylupo 12-0
- Callum Connor 8-0
- Damian Piwowarczyk 11-4
- Emilio Quissua 8-0

## Post-global-database fights

The historical global source ends on 2026-01-31. `week7_postcutoff_fights.csv` records the known subsequent professional bout for each fighter whose latest fight occurred after that cutoff.

These rows are **state refresh inputs**, not DWCS training labels.

## Known prior DWCS evidence

Theo Haig has a prior DWCS appearance: he lost by first-round TKO to Cezary Oleksiejczuk in September 2025. That fight belongs in his prior-DWCS state for Week 7.

## Still required before model lock

1. Generate the full v0.4 fighter snapshots from the global history + current refresh.
2. Verify physical/style fields and leave unknown reach/stance values missing rather than guessing.
3. Freeze prospective tape scores separately from the trained historical v0.4 features.
4. Run `build_current_features.py`.
5. Run `score_current_card.py` before importing sportsbook odds.
