from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

for p in (PROCESSED, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

HISTORY_PATH = RAW / "espn_dwcs_fighter_history_2017_2025.csv"
ATTR_PATH = RAW / "espn_dwcs_fighter_attributes_2017_2025.csv"
STATS_PATH = RAW / "espn_dwcs_fighter_stats_2017_2025.csv"

NON_FEATURE = {
    "event_date", "event_id", "bout_key", "fighter_a_id", "fighter_b_id",
    "fighter_a", "fighter_b", "winner_a", "winner_name", "method",
    "finish_round", "duration_minutes"
}

def safe_num(v):
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan

def ratio(n, d):
    return float(n) / float(d) if d else 0.0

def style_flags(style):
    s = str(style or "").lower()
    return {
        "striker": int(any(k in s for k in ["strik", "boxing", "kickbox", "muay"])),
        "grappler": int(any(k in s for k in ["grap", "wrest", "jiu", "judo", "sambo"])),
    }

def stance_flags(stance):
    s = str(stance or "").lower()
    return {
        "southpaw": int("southpaw" in s),
        "switch": int("switch" in s),
    }

def age_on(dob, event_date):
    if pd.isna(dob):
        return np.nan
    d = pd.Timestamp(dob)
    e = pd.Timestamp(event_date)
    return (e - d).days / 365.2425

def canonical_bout_key(row):
    ids = sorted([str(row.fighter_id), str(row.opponent_id)])
    return f"{row.event_id}|{ids[0]}|{ids[1]}"

def orientation_for(key, ids):
    ordered = sorted(ids)
    parity = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 2
    return ordered if parity == 0 else ordered[::-1]

def state_template():
    return {
        "bouts": 0,
        "wins": 0,
        "finishes": 0,
        "ko_wins": 0,
        "sub_wins": 0,
        "dec_wins": 0,
        "duration_sum": 0.0,
        "over25_sum": 0,
        "stat_bouts": 0,
        "sig_landed": 0.0,
        "sig_attempted": 0.0,
        "td_landed": 0.0,
        "td_attempted": 0.0,
        "knockdowns": 0.0,
        "sub_attempts": 0.0,
        "stat_minutes": 0.0,
    }

def state_features(s):
    bouts = s["bouts"]
    mins = s["stat_minutes"]
    return {
        "prior_dwcs_bouts": float(bouts),
        "prior_dwcs_wins": float(s["wins"]),
        "prior_dwcs_win_pct": ratio(s["wins"], bouts),
        "prior_dwcs_finish_pct": ratio(s["finishes"], bouts),
        "prior_dwcs_ko_win_pct": ratio(s["ko_wins"], bouts),
        "prior_dwcs_sub_win_pct": ratio(s["sub_wins"], bouts),
        "prior_dwcs_dec_win_pct": ratio(s["dec_wins"], bouts),
        "prior_dwcs_avg_duration": ratio(s["duration_sum"], bouts),
        "prior_dwcs_over25_rate": ratio(s["over25_sum"], bouts),
        "prior_dwcs_stats_available": float(s["stat_bouts"] > 0),
        "prior_dwcs_sig_landed_pm": ratio(s["sig_landed"], mins),
        "prior_dwcs_sig_attempted_pm": ratio(s["sig_attempted"], mins),
        "prior_dwcs_sig_accuracy": ratio(s["sig_landed"], s["sig_attempted"]),
        "prior_dwcs_td_landed_per15": 15.0 * ratio(s["td_landed"], mins),
        "prior_dwcs_td_attempted_per15": 15.0 * ratio(s["td_attempted"], mins),
        "prior_dwcs_td_accuracy": ratio(s["td_landed"], s["td_attempted"]),
        "prior_dwcs_kd_per15": 15.0 * ratio(s["knockdowns"], mins),
        "prior_dwcs_sub_attempts_per15": 15.0 * ratio(s["sub_attempts"], mins),
    }

def add_state_result(s, row, stats_row):
    result = str(row.fight_result)
    method = str(row.fight_result_type)
    duration = safe_num(row.fight_duration)

    s["bouts"] += 1
    if result == "W":
        s["wins"] += 1
        if method in {"KO-TKO", "SUBMISSION"}:
            s["finishes"] += 1
        if method == "KO-TKO":
            s["ko_wins"] += 1
        elif method == "SUBMISSION":
            s["sub_wins"] += 1
        elif method.startswith("DEC"):
            s["dec_wins"] += 1

    if np.isfinite(duration):
        s["duration_sum"] += duration
    try:
        s["over25_sum"] += int(bool(row.over_2_5))
    except Exception:
        pass

    if stats_row is not None and np.isfinite(duration) and duration > 0:
        s["stat_bouts"] += 1
        s["stat_minutes"] += duration
        for src, dst in [
            ("SSL", "sig_landed"),
            ("SSA", "sig_attempted"),
            ("TDL", "td_landed"),
            ("TDA", "td_attempted"),
            ("KD", "knockdowns"),
            ("SM", "sub_attempts"),
        ]:
            s[dst] += safe_num(stats_row.get(src, 0)) if np.isfinite(safe_num(stats_row.get(src, 0))) else 0.0

def build_matrix():
    hist = pd.read_csv(HISTORY_PATH)
    attrs = pd.read_csv(ATTR_PATH)
    stats = pd.read_csv(STATS_PATH)

    hist["event_date"] = pd.to_datetime(hist["event_date"])
    attrs["dob"] = pd.to_datetime(attrs["dob"], errors="coerce")
    attr_map = {str(r.fighter_id): r for _, r in attrs.iterrows()}
    stats_map = {str(r.uid): r for _, r in stats.iterrows()}

    hist["bout_key"] = hist.apply(canonical_bout_key, axis=1)
    bouts = []
    ambiguous = 0
    incomplete = 0

    for key, g in hist.groupby("bout_key", sort=False):
        if len(g) != 2:
            incomplete += 1
            continue
        results = set(g.fight_result.astype(str))
        if not ("W" in results and "L" in results):
            ambiguous += 1
            continue
        bouts.append((key, g.copy()))

    bouts.sort(key=lambda kg: (kg[1].event_date.iloc[0], str(kg[1].event_id.iloc[0]), kg[0]))

    state = defaultdict(state_template)
    rows = []

    for key, g in bouts:
        ids = [str(x) for x in g.fighter_id.tolist()]
        a_id, b_id = orientation_for(key, ids)
        by_id = {str(r.fighter_id): r for _, r in g.iterrows()}
        ra, rb = by_id[a_id], by_id[b_id]

        aa, ab = attr_map.get(a_id), attr_map.get(b_id)
        event_date = ra.event_date

        def static(fid, ar):
            sf = style_flags(ar["style"] if ar is not None else "")
            st = stance_flags(ar["stance"] if ar is not None else "")
            return {
                "age": age_on(ar["dob"], event_date) if ar is not None else np.nan,
                "height": safe_num(ar["height"]) if ar is not None else np.nan,
                "reach": safe_num(ar["reach"]) if ar is not None else np.nan,
                "height_missing": float(ar is None or pd.isna(ar["height"])),
                "reach_missing": float(ar is None or pd.isna(ar["reach"])),
                "southpaw": float(st["southpaw"]),
                "switch": float(st["switch"]),
                "striker_style": float(sf["striker"]),
                "grappler_style": float(sf["grappler"]),
            }

        sa, sb = static(a_id, aa), static(b_id, ab)
        pa, pb = state_features(state[a_id]), state_features(state[b_id])

        winner_row = g[g.fight_result.astype(str) == "W"].iloc[0]
        winner_id = str(winner_row.fighter_id)

        row = {
            "event_date": event_date.strftime("%Y-%m-%d"),
            "event_id": str(ra.event_id),
            "bout_key": key,
            "fighter_a_id": a_id,
            "fighter_b_id": b_id,
            "fighter_a": str(ra.fighter_name),
            "fighter_b": str(rb.fighter_name),
            "winner_a": int(winner_id == a_id),
            "winner_name": str(winner_row.fighter_name),
            "method": str(winner_row.fight_result_type),
            "finish_round": int(winner_row.fight_end_round),
            "duration_minutes": safe_num(winner_row.fight_duration),
        }

        for feat in sa:
            row[f"{feat}_diff"] = sa[feat] - sb[feat] if np.isfinite(sa[feat]) and np.isfinite(sb[feat]) else np.nan
        for feat in pa:
            row[f"{feat}_diff"] = pa[feat] - pb[feat]

        rows.append(row)

        # Update both fighter states only after the pre-fight row has been frozen.
        for _, r in g.iterrows():
            fid = str(r.fighter_id)
            add_state_result(state[fid], r, stats_map.get(str(r.uid)))

    matrix = pd.DataFrame(rows)
    meta = {
        "raw_history_rows": int(len(hist)),
        "raw_attribute_rows": int(len(attrs)),
        "raw_stats_rows": int(len(stats)),
        "canonical_bouts": int(hist.bout_key.nunique()),
        "usable_win_loss_bouts": int(len(matrix)),
        "ambiguous_bouts_removed": int(ambiguous),
        "incomplete_bouts_removed": int(incomplete),
        "event_count": int(matrix.event_id.nunique()),
        "fighter_count": int(len(set(matrix.fighter_a_id) | set(matrix.fighter_b_id))),
        "date_min": str(matrix.event_date.min()),
        "date_max": str(matrix.event_date.max()),
        "target_rate_winner_a": float(matrix.winner_a.mean()),
    }
    return matrix, meta

def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]

def make_champion():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.35, penalty="l2", solver="lbfgs", max_iter=5000)),
    ])

def make_challenger():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_depth=3,
            max_iter=250,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

def score(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.c_[1-p, p], labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
    }

def walk_forward(df, features, min_train_events=20):
    d = df.copy()
    d["event_date_dt"] = pd.to_datetime(d.event_date)
    events = (
        d[["event_date_dt", "event_id"]]
        .drop_duplicates()
        .sort_values(["event_date_dt", "event_id"])
        .reset_index(drop=True)
    )

    folds = []
    champ_y, champ_p = [], []
    chall_y, chall_p = [], []

    for i in range(min_train_events, len(events)):
        event_id = str(events.iloc[i].event_id)
        cutoff = events.iloc[i].event_date_dt
        train = d[d.event_date_dt < cutoff].copy()
        test = d[d.event_id.astype(str) == event_id].copy()
        if len(train) < 50 or len(test) == 0 or train.winner_a.nunique() < 2:
            continue

        champion = make_champion()
        challenger = make_challenger()
        champion.fit(train[features], train.winner_a)
        challenger.fit(train[features], train.winner_a)

        pc = champion.predict_proba(test[features])[:, 1]
        pg = challenger.predict_proba(test[features])[:, 1]
        y = test.winner_a.to_numpy()

        champ_y.extend(y.tolist()); champ_p.extend(pc.tolist())
        chall_y.extend(y.tolist()); chall_p.extend(pg.tolist())

        cs, gs = score(y, pc), score(y, pg)
        folds.append({
            "event_date": str(test.event_date.iloc[0]),
            "event_id": event_id,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "champion_accuracy": cs["accuracy"],
            "champion_brier": cs["brier"],
            "champion_log_loss": cs["log_loss"],
            "challenger_accuracy": gs["accuracy"],
            "challenger_brier": gs["brier"],
            "challenger_log_loss": gs["log_loss"],
        })

    summary = {
        "min_train_events": min_train_events,
        "fold_count": len(folds),
        "champion": score(champ_y, champ_p),
        "challenger": score(chall_y, chall_p),
    }
    return pd.DataFrame(folds), summary

def coverage_report(df, features):
    rows = []
    for f in features:
        s = df[f]
        rows.append({
            "feature": f,
            "non_null_pct": float(s.notna().mean()),
            "non_zero_pct": float((s.fillna(0) != 0).mean()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std()) if s.notna().sum() > 1 else None,
        })
    return pd.DataFrame(rows)

def main():
    matrix, meta = build_matrix()
    matrix.to_csv(PROCESSED / "dwcs_historical_prefight_matrix_2017_2025.csv", index=False)

    features = feature_columns(matrix)
    coverage_report(matrix, features).to_csv(REPORTS / "historical_feature_coverage.csv", index=False)

    folds, summary = walk_forward(matrix, features, min_train_events=20)
    folds.to_csv(REPORTS / "winner_walk_forward_folds.csv", index=False)

    champion = make_champion()
    challenger = make_challenger()
    champion.fit(matrix[features], matrix.winner_a)
    challenger.fit(matrix[features], matrix.winner_a)

    # Standardized logistic coefficients are directly interpretable in feature SD units.
    coef = pd.DataFrame({
        "feature": features,
        "coefficient_standardized": champion.named_steps["model"].coef_[0],
    }).sort_values("coefficient_standardized", key=lambda s: s.abs(), ascending=False)
    coef.to_csv(REPORTS / "winner_logistic_coefficients.csv", index=False)

    joblib.dump(champion, MODELS / "dwcs_w_logistic_v0_3.joblib")
    joblib.dump(challenger, MODELS / "dwcs_w_hgb_v0_3.joblib")

    # Promotion rule: challenger must improve Brier and log loss, not just accuracy.
    c = summary["champion"]; g = summary["challenger"]
    if g["brier"] < c["brier"] and g["log_loss"] < c["log_loss"]:
        recommendation = "challenger"
    else:
        recommendation = "champion"

    report = {
        "model_version": "DWCS-W-v0.3-historical-baseline",
        "source_scope": "DWCS only, ESPN/Kaggle mirror through 2025",
        "important_limitation": (
            "This baseline has physical attributes and prior-DWCS history/stats, "
            "but not full pre-DWCS regional career records for debutants. "
            "It is a real leakage-safe baseline, not the final career-strength model."
        ),
        "matrix": meta,
        "features": features,
        "walk_forward": summary,
        "promotion_recommendation": recommendation,
        "week6_feature_contract_status": (
            "Tape-derived Week 6 lesson features are defined in docs/feature_contract.md "
            "but are not retroactively invented for historical fights."
        ),
    }
    (REPORTS / "winner_walk_forward_summary.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
