# Current Card Intake — DWCS-W v0.4

This workflow is for Week 7 and every card after it.

## 1. Freeze fighter snapshots

Create one row per fighter from `data/current/fighter_snapshot_template.csv`.

The snapshot must reflect information known **before** the event. It contains:

- physical/style fields used by the historical DWCS model;
- prior-DWCS performance fields;
- current career record and recent 3/5 form;
- KO/SUB/decision mix and finish-loss rate;
- opponent-strength estimates;
- Elo / major-organization experience;
- win streak and inactivity;
- a provenance note and as-of date.

Set `history_established=1` only when the current pre-fight regional/career history has actually been established. Do not mark it 1 just because a fighter has a listed W-L record.

## 2. Define matchups

Fill `data/current/card_matchups_template.csv`.

## 3. Build model feature differences

```bash
python src/build_current_features.py \
  --snapshots data/current/week7_fighter_snapshots.csv \
  --matchups data/current/week7_matchups.csv \
  --out data/current/week7_model_features.csv
```

## 4. Score and freeze before odds

```bash
python src/score_current_card.py \
  --features data/current/week7_model_features.csv \
  --out official_picks/week7_winner_model_lock.csv \
  --receipt official_picks/week7_winner_model_receipt.json
```

The scorer rejects odds/sportsbook columns. It uses the exact audited
`models/dwcs_w_v0_4_promoted_logistic.joblib` artifact and then applies the
current weekly calibration layer.

## Coverage gate

v0.4 is primary only if **both fighters have established current career histories**.

If the gate fails, do not silently impute a prospect's whole regional résumé.
Use the v0.3 fallback plus current tape/research until the missing history can be established.

## Tape lessons

Week 6 tape-derived lessons remain separate from the historical v0.4 trained
features because they were not honestly backfilled across old fights. Save them
as prospective research annotations; do not invent historical labels to make the
model look more sophisticated.
