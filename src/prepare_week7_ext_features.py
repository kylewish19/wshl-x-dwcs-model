from __future__ import annotations

from pathlib import Path
import math

import duckdb
import kagglehub
import numpy as np
import pandas as pd

from enrich_with_global_career import (
    SNAPSHOT_FEATURES, build_histories, norm_name, snapshot, expected, ELO_K
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data" / "current"
MATCHUPS = CURRENT / "week7_matchups.csv"
BASE_FEATURES = CURRENT / "week7_model_features.csv"
POSTCUT = CURRENT / "week7_postcutoff_fights.csv"
OUT = CURRENT / "week7_secondary_model_features.csv"


def latest_elo_before(records, cutoff):
    if not records:
        return 1500.0
    eligible = [r for r in records if pd.Timestamp(r["date"]) < pd.Timestamp(cutoff)]
    return float(eligible[-1]["post_elo"]) if eligible else 1500.0


def normalize_method(m):
    s = str(m or "").upper()
    if "KO" in s or "TKO" in s:
        return "KO_TKO"
    if "SUB" in s:
        return "SUB"
    if "DEC" in s:
        return "DEC"
    return "OTHER"


def append_postcut(histories, postcut):
    # The global source currently stops before these 2026 bouts. Add only the
    # known prospect-side result; do not fabricate duration, round, title status,
    # or opponent-side career updates.
    for r in postcut.sort_values("event_date").itertuples(index=False):
        name = norm_name(r.fighter_name)
        opp = norm_name(r.opponent_name)
        date = pd.Timestamp(r.event_date)
        recs = histories.setdefault(name, [])

        pre_elo = latest_elo_before(recs, date)
        opp_elo = latest_elo_before(histories.get(opp, []), date)
        won = str(r.result).upper() == "W"
        score = 1.0 if won else 0.0
        post_elo = pre_elo + ELO_K * (score - expected(pre_elo, opp_elo))

        recs.append({
            "date": date,
            "method": normalize_method(r.method),
            "major": bool(int(r.is_major_org)),
            "org": "POSTCUT_CURRENT_RESEARCH",
            "title": False,
            "duration": np.nan,
            "round": None,
            "result": "W" if won else "L",
            "opp_elo": float(opp_elo),
            "post_elo": float(post_elo),
        })
        recs.sort(key=lambda x: x["date"])


def main():
    matchups = pd.read_csv(MATCHUPS)
    base = pd.read_csv(BASE_FEATURES)
    postcut = pd.read_csv(POSTCUT)
    event_date = pd.Timestamp(matchups.event_date.iloc[0])

    data_dir = Path(kagglehub.dataset_download("leandroiber/mmastats"))
    dbs = list(data_dir.rglob("*.duckdb"))
    if not dbs:
        raise FileNotFoundError("No mmastats DuckDB found")

    con = duckdb.connect(str(dbs[0]), read_only=True)
    global_df = con.execute("""
        SELECT fight_id, organization, event_date, fighter_1, fighter_2, winner,
               method_normalized, round_num, time_finish_seconds,
               is_major_org, is_title_fight
        FROM fights_career_longitudinal
        WHERE event_date < CAST(? AS DATE)
        ORDER BY event_date, fight_id
    """, [str(event_date.date())]).fetchdf()
    con.close()

    histories = build_histories(global_df)
    append_postcut(histories, postcut)

    snapshots = {}
    for name in sorted(set(matchups.fighter_a) | set(matchups.fighter_b)):
        snapshots[norm_name(name)] = snapshot(histories.get(norm_name(name), []), event_date)

    rows = []
    for b in matchups.itertuples(index=False):
        a = snapshots[norm_name(b.fighter_a)]
        bb = snapshots[norm_name(b.fighter_b)]
        row = {}
        for feat in SNAPSHOT_FEATURES:
            va, vb = a[feat], bb[feat]
            if np.isfinite(va) and np.isfinite(vb):
                row[f"{feat}_diff"] = float(va - vb)
            else:
                row[f"{feat}_diff"] = np.nan
        row["ext_a_available"] = float(a["ext_available"])
        row["ext_b_available"] = float(bb["ext_available"])
        rows.append(row)

    ext = pd.DataFrame(rows)
    out = pd.concat([base.reset_index(drop=True), ext.reset_index(drop=True)], axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("Wrote", OUT)
    print("rows", len(out), "columns", len(out.columns))
    print("external availability", out[["ext_a_available","ext_b_available"]].to_dict("records"))


if __name__ == "__main__":
    main()
