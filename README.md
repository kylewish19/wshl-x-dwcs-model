# WSHL_X DWCS Model

A **Dana White's Contender Series-only** prediction and grading system.

## Scope

This repository is intentionally separate from the UFC model. It tracks and updates DWCS models for:

- fight winner
- method of victory (KO/TKO, submission, decision)
- finish round
- round totals / O-U markets
- probability calibration
- 10,000-fight Monte Carlo output

## Weekly workflow

1. Freeze all pre-fight features.
2. Train/retrain using only fights that have already happened.
3. Lock the card before sportsbook-value comparison.
4. Save the locked probabilities and model version.
5. Grade every prediction after the card.
6. Append results to the training ledger.
7. Retrain and record coefficient / calibration changes.
8. Promote lessons only when they can be converted into measurable features.

## Anti-leakage rules

- No Week N result may train the Week N model.
- No sportsbook odds are model features.
- No UFC fights are training rows.
- Historical features must be frozen to information available before the bout.
- PASS is a valid model outcome.
- Monte Carlo consumes fitted probabilities; it is not a substitute for a trained predictor.

## Current state

The recoverable Weeks 2-5 block was 12-8 on winners but was produced by a mixed/manual workflow. Week 6 was the first prospectively locked card of the rebuild and graded:

- Winners: 4-1
- Most-likely method: 4-1
- Round / decision bucket: 4-1
- O/U 1.5 direction: 5-0
- O/U 2.5 direction: 5-0

Current rebuild status:

- DWCS calibration layer: working and retrained weekly
- DWCS method population prior: working and retrained weekly
- DWCS O/U 1.5 population prior: working and retrained weekly
- Fighter-level winner engine: code-backed, historical feature matrix still being completed
- Exact-round fighter-level model: not yet promoted

The goal is not to pretend an unfinished component is production-ready. Every model version and weekly card should be auditable from this repository.
