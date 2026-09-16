# External / Regional Career History Policy

The DWCS model may use external professional MMA history as **fighter state**, but external fights are never DWCS training labels.

## Allowed uses

External bouts that occurred before a DWCS bout may update:

- career fight count
- career win percentage
- career finish percentage
- KO/TKO win rate
- submission win rate
- finish-loss vulnerability
- recent-3 and recent-5 form
- days since last fight
- opponent-strength state
- Elo-style strength rating
- average prior opponent Elo

## Anti-leakage

For a DWCS fight on date D, only external bouts strictly before D may affect that row.

No current or future fight outcome may be used to reconstruct a pre-fight feature.

## Source policy

Current bootstrap source:

- Kaggle: binduvr/pro-mma-fights
- publisher-declared license: CC0
- historical coverage: UFC, Bellator, ONE through August 2021
- original provenance: scraped from Sherdog by the dataset publisher

This bootstrap does **not** represent a complete regional career ledger. It is therefore treated as a challenger feature layer, not automatically promoted.

## Promotion gate

An external-history feature layer may be promoted only when chronological walk-forward evaluation on the same DWCS holdout rows improves BOTH:

1. Brier score
2. log loss

Accuracy alone is insufficient.

## Missing external history

No external history does not mean zero skill.

Missing external history must be represented explicitly through availability indicators. Missing strike/takedown/control data may not be imputed as observed zeroes.

## Future upgrade

The preferred future source is a licensed/permissioned comprehensive professional-MMA record feed capable of covering regional promotions used by DWCS prospects. When such a source is available, its exact snapshot/hash and licensing provenance should be pinned in the model registry before production use.
