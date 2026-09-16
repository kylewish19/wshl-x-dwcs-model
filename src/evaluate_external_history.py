from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_2017_2025.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

BASE_NON_FEATURE = {
    "event_date", "event_id", "bout_key", "fighter_a_id", "fighter_b_id",
    "fighter_a", "fighter_b", "winner_a", "winner_name", "method",
    "finish_round", "duration_minutes"
}


def norm_name(x: str) -> str:
    s = unicodedata.normalize("NFKD", str(x or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def parse_date(x):
    return pd.to_datetime(x, errors="coerce")


def is_finish(method: str) -> bool:
    s = str(method or "").lower()
    return any(k in s for k in ["ko", "tko", "submission"])


def method_kind(method: str) -> str:
    s = str(method or "").lower()
    if "submission" in s:
        return "sub"
    if "ko" in s or "tko" in s:
        return "ko"
    return "other"


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def default_state():
    return {
        "elo": 1500.0,
        "fights": 0,
        "wins": 0,
        "losses": 0,
        "fin_wins": 0,
        "ko_wins": 0,
        "sub_wins": 0,
        "fin_losses": 0,
        "opp_elo_sum": 0.0,
        "last_date": None,
        "recent": deque(maxlen=5),
    }


def state_snapshot(s, date):
    fights = s["fights"]
    recent = list(s["recent"])
    recent3 = recent[-3:]
    days = np.nan if s["last_date"] is None else (date - s["last_date"]).days
    return {
        "ext_fights": float(fights),
        "ext_win_pct": s["wins"] / fights if fights else 0.0,
        "ext_finish_win_pct": s["fin_wins"] / fights if fights else 0.0,
        "ext_ko_win_pct": s["ko_wins"] / fights if fights else 0.0,
        "ext_sub_win_pct": s["sub_wins"] / fights if fights else 0.0,
        "ext_finish_loss_pct": s["fin_losses"] / fights if fights else 0.0,
        "ext_recent3_win_pct": np.mean(recent3) if recent3 else 0.0,
        "ext_recent5_win_pct": np.mean(recent) if recent else 0.0,
        "ext_elo": s["elo"],
        "ext_avg_opp_elo": s["opp_elo_sum"] / fights if fights else 1500.0,
        "ext_days_since_last": float(days) if pd.notna(days) else np.nan,
        "ext_history_available": float(fights > 0),
    }


def replay_external(external: pd.DataFrame, snapshots_needed: dict[str, list[pd.Timestamp]]):
    external = external.copy()
    external["date"] = external["date"].map(parse_date)
    external = external.dropna(subset=["date"]).sort_values(["date", "event_title", "match_nr"])

    states = defaultdict(default_state)
    snapshots = defaultdict(dict)
    dates_by_name = {n: sorted(set(ds)) for n, ds in snapshots_needed.items()}
    idx = {n: 0 for n in dates_by_name}

    def flush_until(current_date):
        for n, dates in dates_by_name.items():
            i = idx[n]
            while i < len(dates) and dates[i] <= current_date:
                snapshots[n][dates[i]] = state_snapshot(states[n], dates[i])
                i += 1
            idx[n] = i

    for _, r in external.iterrows():
        d = r["date"]
        flush_until(d)
        a, b = norm_name(r["fighter1_name"]), norm_name(r["fighter2_name"])
        if not a or not b:
            continue
        sa, sb = states[a], states[b]
        ra, rb = sa["elo"], sb["elo"]
        a_win = str(r["fighter1_result"]).lower() == "win"
        b_win = str(r["fighter2_result"]).lower() == "win"
        if not (a_win ^ b_win):
            continue
        score_a = 1.0 if a_win else 0.0
        ea = expected(ra, rb)
        k = 32.0
        sa["elo"] = ra + k * (score_a - ea)
        sb["elo"] = rb + k * ((1.0 - score_a) - (1.0 - ea))
        kind = method_kind(r.get("win_method", ""))
        for s, won, opp_rating in [(sa, a_win, rb), (sb, b_win, ra)]:
            s["fights"] += 1
            s["wins"] += int(won)
            s["losses"] += int(not won)
            s["opp_elo_sum"] += opp_rating
            s["recent"].append(1 if won else 0)
            if won and is_finish(r.get("win_method", "")):
                s["fin_wins"] += 1
            if (not won) and is_finish(r.get("win_method", "")):
                s["fin_losses"] += 1
            if won and kind == "ko":
                s["ko_wins"] += 1
            if won and kind == "sub":
                s["sub_wins"] += 1
            s["last_date"] = d

    flush_until(pd.Timestamp.max - pd.Timedelta(days=2))
    return snapshots


def score(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.c_[1-p, p], labels=[0,1])),
        "auc": float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else None,
    }


def model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.35, penalty="l2", solver="lbfgs", max_iter=5000)),
    ])


def evaluate(df, base_features, aug_features, min_train_events=12):
    d = df.sort_values(["event_date", "event_id"]).copy()
    events = d[["event_date", "event_id"]].drop_duplicates().sort_values(["event_date","event_id"]).reset_index(drop=True)
    by, bp, ay, ap = [], [], [], []
    folds=[]
    for i in range(min_train_events, len(events)):
        eid = events.iloc[i].event_id
        cutoff = events.iloc[i].event_date
        train = d[d.event_date < cutoff]
        test = d[d.event_id == eid]
        if len(train) < 50 or len(test) == 0 or train.winner_a.nunique() < 2:
            continue
        mb, ma = model(), model()
        mb.fit(train[base_features], train.winner_a)
        ma.fit(train[aug_features], train.winner_a)
        p0 = mb.predict_proba(test[base_features])[:,1]
        p1 = ma.predict_proba(test[aug_features])[:,1]
        y = test.winner_a.to_numpy()
        by.extend(y); bp.extend(p0); ay.extend(y); ap.extend(p1)
        folds.append({
            "event_date": str(test.event_date.iloc[0].date()),
            "event_id": str(eid),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "base_brier": score(y,p0)["brier"],
            "aug_brier": score(y,p1)["brier"],
            "base_log_loss": score(y,p0)["log_loss"],
            "aug_log_loss": score(y,p1)["log_loss"],
        })
    return pd.DataFrame(folds), {"baseline": score(by,bp), "augmented": score(ay,ap)}


def main(external_csv):
    matrix = pd.read_csv(MATRIX)
    matrix["event_date"] = pd.to_datetime(matrix["event_date"])
    ext = pd.read_csv(external_csv)
    cutoff = pd.to_datetime(ext["date"], errors="coerce").max()

    # Only evaluate rows whose outcomes occur within external-source coverage.
    eval_df = matrix[matrix.event_date <= cutoff].copy()
    needed = defaultdict(list)
    for _, r in eval_df.iterrows():
        needed[norm_name(r.fighter_a)].append(r.event_date)
        needed[norm_name(r.fighter_b)].append(r.event_date)
    snaps = replay_external(ext, needed)

    ext_fields = [
        "ext_fights", "ext_win_pct", "ext_finish_win_pct", "ext_ko_win_pct",
        "ext_sub_win_pct", "ext_finish_loss_pct", "ext_recent3_win_pct",
        "ext_recent5_win_pct", "ext_elo", "ext_avg_opp_elo",
        "ext_days_since_last", "ext_history_available"
    ]
    rows=[]
    for _, r in eval_df.iterrows():
        a, b = norm_name(r.fighter_a), norm_name(r.fighter_b)
        sa = snaps.get(a, {}).get(r.event_date, state_snapshot(default_state(), r.event_date))
        sb = snaps.get(b, {}).get(r.event_date, state_snapshot(default_state(), r.event_date))
        rr = r.to_dict()
        for f in ext_fields:
            va, vb = sa[f], sb[f]
            rr[f+"_diff"] = (va - vb) if pd.notna(va) and pd.notna(vb) else np.nan
        rr["ext_any_history"] = float(sa["ext_history_available"] or sb["ext_history_available"])
        rr["ext_both_history"] = float(sa["ext_history_available"] and sb["ext_history_available"])
        rows.append(rr)
    out = pd.DataFrame(rows)

    numeric = [c for c in out.columns if c not in BASE_NON_FEATURE and pd.api.types.is_numeric_dtype(out[c])]
    external_features = [c for c in numeric if c.startswith("ext_")]
    base_features = [c for c in numeric if c not in external_features]
    aug_features = base_features + external_features

    # Main audit: rows where at least one fighter has external-history state.
    aud = out[out.ext_any_history == 1].copy()
    folds, summary = evaluate(aud, base_features, aug_features)
    summary.update({
        "external_source_cutoff": str(cutoff.date()),
        "rows_in_source_window": int(len(out)),
        "rows_with_any_external_history": int((out.ext_any_history==1).sum()),
        "rows_with_both_external_history": int((out.ext_both_history==1).sum()),
        "external_features": external_features,
        "promotion_rule": "Promote only if augmented Brier and log loss are both lower than baseline on the same walk-forward rows.",
    })
    summary["promotion_recommendation"] = (
        "promote_external_layer"
        if summary["augmented"]["brier"] < summary["baseline"]["brier"]
        and summary["augmented"]["log_loss"] < summary["baseline"]["log_loss"]
        else "do_not_promote"
    )

    folds.to_csv(REPORTS / "external_history_walk_forward_folds.csv", index=False)
    (REPORTS / "external_history_walk_forward_summary.json").write_text(json.dumps(summary, indent=2))
    out.to_csv(ROOT / "data" / "processed" / "dwcs_external_history_audit_matrix.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("external_csv")
    args = ap.parse_args()
    main(args.external_csv)
