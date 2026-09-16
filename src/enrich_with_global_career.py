from __future__ import annotations

import bisect
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb
import kagglehub
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_2017_2025.csv"
OUT = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_enriched_2017_2025.csv"
REPORT = ROOT / "reports" / "global_career_enrichment_report.json"

ELO_K = 24.0
ELO_START = 1500.0


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def family(method):
    s = str(method or "").lower()
    if s in {"ko", "tko"}:
        return "KO_TKO"
    if "submission" in s:
        return "SUB"
    if "decision" in s:
        return "DEC"
    return "OTHER"


def total_seconds(round_num, round_seconds):
    try:
        r = int(round_num)
        s = int(round_seconds)
        if r < 1 or s < 0:
            return np.nan
        return (r - 1) * 300 + s
    except Exception:
        return np.nan


def expected(ra, rb):
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def build_histories(global_df):
    ratings = defaultdict(lambda: ELO_START)
    hist = defaultdict(list)

    g = global_df.sort_values(["event_date", "fight_id"]).copy()

    for row in g.itertuples(index=False):
        n1 = norm_name(row.fighter_1)
        n2 = norm_name(row.fighter_2)
        if not n1 or not n2 or n1 == n2:
            continue

        r1, r2 = ratings[n1], ratings[n2]
        win = norm_name(row.winner)
        if win == n1:
            s1, s2 = 1.0, 0.0
            res1, res2 = "W", "L"
        elif win == n2:
            s1, s2 = 0.0, 1.0
            res1, res2 = "L", "W"
        else:
            s1 = s2 = 0.5
            res1 = res2 = "D"

        e1, e2 = expected(r1, r2), expected(r2, r1)
        post1 = r1 + ELO_K * (s1 - e1)
        post2 = r2 + ELO_K * (s2 - e2)

        meth = family(row.method_normalized)
        dur = total_seconds(row.round_num, row.time_finish_seconds)
        common = {
            "date": pd.Timestamp(row.event_date),
            "method": meth,
            "major": bool(row.is_major_org) if pd.notna(row.is_major_org) else False,
            "org": str(row.organization),
            "title": bool(row.is_title_fight) if pd.notna(row.is_title_fight) else False,
            "duration": dur,
            "round": int(row.round_num) if pd.notna(row.round_num) else None,
        }
        hist[n1].append({
            **common, "result": res1, "opp_elo": r2, "post_elo": post1,
        })
        hist[n2].append({
            **common, "result": res2, "opp_elo": r1, "post_elo": post2,
        })
        ratings[n1], ratings[n2] = post1, post2

    # Lists are already chronological because g was sorted.
    return hist


SNAPSHOT_FEATURES = [
    "ext_available",
    "ext_fights",
    "ext_wins",
    "ext_losses",
    "ext_win_pct",
    "ext_finish_win_pct",
    "ext_ko_win_pct",
    "ext_sub_win_pct",
    "ext_dec_win_pct",
    "ext_finish_loss_pct",
    "ext_ko_loss_pct",
    "ext_sub_loss_pct",
    "ext_recent3_win_pct",
    "ext_recent5_win_pct",
    "ext_recent3_finish_pct",
    "ext_recent5_finish_pct",
    "ext_major_fights",
    "ext_major_win_pct",
    "ext_regional_fights",
    "ext_regional_win_pct",
    "ext_org_diversity",
    "ext_title_fights",
    "ext_elo",
    "ext_avg_opp_elo",
    "ext_recent5_avg_opp_elo",
    "ext_best_win_opp_elo",
    "ext_days_since_last",
    "ext_avg_duration_sec",
    "ext_recent5_avg_duration_sec",
    "ext_r1_finish_win_pct",
    "ext_early_finish_win_pct",
    "ext_multi_route_finisher",
]


def _rate(num, den):
    return float(num) / float(den) if den else 0.0


def snapshot(records, cutoff):
    if not records:
        return {f: (1500.0 if f == "ext_elo" else np.nan if f == "ext_days_since_last" else 0.0)
                for f in SNAPSHOT_FEATURES}

    dates = [r["date"] for r in records]
    i = bisect.bisect_left(dates, pd.Timestamp(cutoff))
    recs = records[:i]
    if not recs:
        return {f: (1500.0 if f == "ext_elo" else np.nan if f == "ext_days_since_last" else 0.0)
                for f in SNAPSHOT_FEATURES}

    wins = [r for r in recs if r["result"] == "W"]
    losses = [r for r in recs if r["result"] == "L"]
    recent3 = recs[-3:]
    recent5 = recs[-5:]
    finish_methods = {"KO_TKO", "SUB"}

    def win_rate(rs):
        return _rate(sum(r["result"] == "W" for r in rs), len(rs))

    def finish_rate_wins(rs):
        wr = [r for r in rs if r["result"] == "W"]
        return _rate(sum(r["method"] in finish_methods for r in wr), len(wr))

    majors = [r for r in recs if r["major"]]
    regionals = [r for r in recs if not r["major"]]
    durations = [r["duration"] for r in recs if np.isfinite(r["duration"])]
    recent_durations = [r["duration"] for r in recent5 if np.isfinite(r["duration"])]
    win_opp = [r["opp_elo"] for r in wins]

    ko_w = sum(r["method"] == "KO_TKO" for r in wins)
    sub_w = sum(r["method"] == "SUB" for r in wins)
    dec_w = sum(r["method"] == "DEC" for r in wins)
    fin_l = sum(r["method"] in finish_methods for r in losses)
    ko_l = sum(r["method"] == "KO_TKO" for r in losses)
    sub_l = sum(r["method"] == "SUB" for r in losses)
    r1_fin_w = sum((r["method"] in finish_methods and r["round"] == 1) for r in wins)
    early_fin_w = sum((r["method"] in finish_methods and np.isfinite(r["duration"]) and r["duration"] < 450) for r in wins)

    return {
        "ext_available": 1.0,
        "ext_fights": float(len(recs)),
        "ext_wins": float(len(wins)),
        "ext_losses": float(len(losses)),
        "ext_win_pct": _rate(len(wins), len(recs)),
        "ext_finish_win_pct": _rate(ko_w + sub_w, len(wins)),
        "ext_ko_win_pct": _rate(ko_w, len(wins)),
        "ext_sub_win_pct": _rate(sub_w, len(wins)),
        "ext_dec_win_pct": _rate(dec_w, len(wins)),
        "ext_finish_loss_pct": _rate(fin_l, len(losses)),
        "ext_ko_loss_pct": _rate(ko_l, len(losses)),
        "ext_sub_loss_pct": _rate(sub_l, len(losses)),
        "ext_recent3_win_pct": win_rate(recent3),
        "ext_recent5_win_pct": win_rate(recent5),
        "ext_recent3_finish_pct": finish_rate_wins(recent3),
        "ext_recent5_finish_pct": finish_rate_wins(recent5),
        "ext_major_fights": float(len(majors)),
        "ext_major_win_pct": win_rate(majors),
        "ext_regional_fights": float(len(regionals)),
        "ext_regional_win_pct": win_rate(regionals),
        "ext_org_diversity": float(len({r["org"] for r in recs})),
        "ext_title_fights": float(sum(r["title"] for r in recs)),
        "ext_elo": float(recs[-1]["post_elo"]),
        "ext_avg_opp_elo": float(np.mean([r["opp_elo"] for r in recs])),
        "ext_recent5_avg_opp_elo": float(np.mean([r["opp_elo"] for r in recent5])),
        "ext_best_win_opp_elo": float(max(win_opp)) if win_opp else 1500.0,
        "ext_days_since_last": float((pd.Timestamp(cutoff) - recs[-1]["date"]).days),
        "ext_avg_duration_sec": float(np.mean(durations)) if durations else np.nan,
        "ext_recent5_avg_duration_sec": float(np.mean(recent_durations)) if recent_durations else np.nan,
        "ext_r1_finish_win_pct": _rate(r1_fin_w, len(wins)),
        "ext_early_finish_win_pct": _rate(early_fin_w, len(wins)),
        "ext_multi_route_finisher": float(ko_w > 0 and sub_w > 0),
    }


def main():
    base = pd.read_csv(BASE)
    base["event_date"] = pd.to_datetime(base.event_date)

    data_dir = Path(kagglehub.dataset_download("leandroiber/mmastats"))
    dbs = list(data_dir.rglob("*.duckdb"))
    if not dbs:
        raise FileNotFoundError(f"No DuckDB file found under {data_dir}")
    db = dbs[0]

    con = duckdb.connect(str(db), read_only=True)
    max_date = pd.Timestamp(base.event_date.max()).strftime("%Y-%m-%d")
    global_df = con.execute(f"""
        SELECT
            fight_id, organization, event_date, fighter_1, fighter_2, winner,
            method_normalized, round_num, time_finish_seconds,
            is_major_org, is_title_fight
        FROM fights_career_longitudinal
        WHERE event_date < DATE '{max_date}'
        ORDER BY event_date, fight_id
    """).fetchdf()
    con.close()

    histories = build_histories(global_df)

    rows = []
    fighter_match_flags = []
    for row in base.itertuples(index=False):
        cutoff = pd.Timestamp(row.event_date)
        na = norm_name(row.fighter_a)
        nb = norm_name(row.fighter_b)
        sa = snapshot(histories.get(na, []), cutoff)
        sb = snapshot(histories.get(nb, []), cutoff)

        d = row._asdict()
        for feat in SNAPSHOT_FEATURES:
            va, vb = sa[feat], sb[feat]
            if np.isfinite(va) and np.isfinite(vb):
                d[f"{feat}_diff"] = float(va - vb)
            else:
                d[f"{feat}_diff"] = np.nan

        d["ext_a_available"] = sa["ext_available"]
        d["ext_b_available"] = sb["ext_available"]
        rows.append(d)
        fighter_match_flags.extend([sa["ext_available"], sb["ext_available"]])

    out = pd.DataFrame(rows)
    out["event_date"] = pd.to_datetime(out.event_date).dt.strftime("%Y-%m-%d")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    both = ((out.ext_a_available == 1) & (out.ext_b_available == 1)).mean()
    one_or_more = ((out.ext_a_available == 1) | (out.ext_b_available == 1)).mean()
    rep = {
        "source": "LeandroIber/mmastats via Kaggle",
        "license": "MIT",
        "source_rows_loaded_before_last_dwcs_date": int(len(global_df)),
        "dwcs_rows": int(len(out)),
        "fighter_slots": int(len(out) * 2),
        "fighter_slot_history_coverage": float(np.mean(fighter_match_flags)),
        "both_fighters_history_coverage": float(both),
        "at_least_one_fighter_history_coverage": float(one_or_more),
        "elo_start": ELO_START,
        "elo_k": ELO_K,
        "feature_count_added": len(SNAPSHOT_FEATURES),
        "anti_leakage": "Only global fights with event_date strictly earlier than each DWCS bout are used.",
        "training_label_scope": "DWCS labels only; global fights are context features only.",
    }
    REPORT.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
