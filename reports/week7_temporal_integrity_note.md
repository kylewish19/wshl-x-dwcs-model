# DWCS Week 7 temporal-integrity note

Event: DWCS Season 10 Week 7  
Event date: 2026-09-22  
Scheduled card start: 7:00 p.m. ET

## Original prospective artifact

The original Week 7 model lock was committed before the event at 2026-09-22T20:20:13Z (4:20 p.m. ET), commit `1dadf18e1201758f913260b97a243497d3fba79a`.

That artifact is genuinely pre-event, but a data-quality issue was later identified in the current-form construction. The mixed-source recent-history path produced impossible or incorrect professional win-streak / recent-form values for several fighters.

Therefore the original Week 7 lock is **prospective but data-quality tainted**. It should not be used as a clean model-validation card.

## Corrected current-form inputs

Professional recent form was reconstructed from current Sherdog pro fight histories and stored in `data/current/week7_current_records.csv`.

Corrected examples:
- Damian Piwowarczyk: recent 3 = 1.00, recent 5 = 0.60, current pro win streak = 3
- Emilio Quissua: recent 3 = 1.00, recent 5 = 1.00, current pro win streak = 8
- Theo Haig: recent 3 = 0.667, recent 5 = 0.80, current pro win streak = 1
- Marcos Degli: current pro win streak = 12

The snapshot builder was changed so reconciled professional current-form inputs override the mixed-source recent-history sequence, with an integrity guard preventing a pro win streak from exceeding total pro wins.

## Corrected rerun

The corrected four-model rerun completed at approximately 2026-09-22T23:19Z (7:19 p.m. ET), after the scheduled event start.

No Week 7 fight result was used to construct the corrected features or probabilities. However, because the rerun occurred after the scheduled start, it is **diagnostic only** and must not be represented as a clean prospective lock.

Corrected diagnostic winner probabilities:
- Norbert Novenyi Jr. 61.94% over Theo Haig
- Alvi Dasuyev 71.21% over Jaden Ortega
- Marcos Degli 94.52% over Paris Moran
- Piero Guaylupo 80.95% over Callum Connor
- Emilio Quissua 82.64% over Damian Piwowarczyk

Diagnostic artifacts:
- `diagnostics/week7_corrected_form_rerun_after_start.csv`
- `diagnostics/week7_corrected_form_rerun_after_start.json`

The original pre-event files were restored to preserve the true prospective record.

## Grading policy

- Do not count the corrected rerun as prospective validation.
- Mark the original Week 7 card as prospective/data-quality-tainted rather than clean.
- Actual Week 7 outcomes may still be added to future training after pre-fight features are reconstructed with the corrected form pipeline.
- Week 8+ locks must use reconciled professional recent-form inputs before the event and must be frozen before the scheduled start.
