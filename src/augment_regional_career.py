from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path

import duckdb
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
BASE_MATRIX = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_2017_2025.csv"
OUT_MATRIX = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_regional_candidate.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
for p in (REPORTS, MODELS, OUT_MATRIX.parent):
    p.mkdir(parents=True, exist_ok=True)

NON_FEATURE = {
    "event_date", "event_id", "bout_key", "fighter_a_id", "fighter_b_id",
    "fighter_a", "fighter_b", "winner_a", "winner_name", "method",
    "finish_round", "duration_minutes",
}
GLOBAL_PREFIX = "global_"

def norm_name(value: object) -> str:
    s = unicodedata.normalize("NFKD", str(value or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().replace("’", "'")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

def ratio(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0

def state_template():
    return {
        "bouts": 0,
        "wins": 0,
        "losses": 0,
        "finish_wins": 0,
        "ko_wins": 0,
        "sub_wins": 0,
        "dec_wins": 0,
        "finish_losses": 0,
        "quality_wins": 0,
        "opp_win_pct_sum": 0.0,
        "opp_elo_sum": 0.0,
        "opp_quality_n": 0,
        "major_bouts": 0,
        "major_wins": 0,
        "win_streak": 0,
        "recent_results": deque(maxlen=5),
        "elo": 1500.0,
        "last_date": None,
    }

def snapshot(s: dict) -> dict:
    decided = s["wins"] + s["losses"]
    recent = list(s["recent_results"])
    recent3 = recent[-3:]
    recent5 = recent[-5:]
    return {
        "history_available": float(s["bouts"] > 0),
        "career_bouts": float(s["bouts"]),
        "career_wins": float(s["wins"]),
        "career_losses": float(s["losses"]),
        "career_win_pct": ratio(s["wins"], decided),
        "recent3_win_pct": ratio(sum(recent3), len(recent3)),
        "recent5_win_pct": ratio(sum(recent5), len(recent5)),
        "career_finish_pct": ratio(s["finish_wins"], s["wins"]),
        "ko_win_pct": ratio(s["ko_wins"], s["wins"]),
        "sub_win_pct": ratio(s["sub_wins"], s["wins"]),
        "decision_win_pct": ratio(s["dec_wins"], s["wins"]),
        "finish_loss_pct": ratio(s["finish_losses"], s["losses"]),
        "avg_opp_win_pct": ratio(s["opp_win_pct_sum"], s["opp_quality_n"]),
        "avg_opp_elo": ratio(s["opp_elo_sum"], s["opp_quality_n"]),
        "quality_win_count": float(s["quality_wins"]),
        "elo": float(s["elo"]),
        "major_bouts": float(s["major_bouts"]),
        "major_win_pct": ratio(s["major_wins"], s["major_bouts"]),
        "win_streak": float(s["win_streak"]),
        "last_date": s["last_date"],
    }

def result_side(winner: object, f1: str, f2: str):
    w = norm_name(winner)
    if not w:
        return None
    if w == f1:
        return 1
    if w == f2:
        return 2
    return None

def is_finish(method: object) -> bool:
    m = str(method or "").casefold()
    return m in {"ko", "tko", "submission"} or "submission" in m or m.startswith("ko") or m.startswith("tko")

def method_bucket(method: object) -> str:
    m = str(method or "").casefold()
    if "submission" in m:
        return "SUB"
    if m.startswith("ko") or m.startswith("tko"):
        return "KO"
    if "decision" in m:
        return "DEC"
    return "OTHER"

def expected(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))

def update_one(s, opp_pre, won: bool | None, method, major: bool, date):
    s["bouts"] += 1
    s["last_date"] = date
    opp_decided = opp_pre["wins"] + opp_pre["losses"]
    opp_wp = ratio(opp_pre["wins"], opp_decided) if opp_decided else 0.5
    s["opp_win_pct_sum"] += opp_wp
    s["opp_elo_sum"] += opp_pre["elo"]
    s["opp_quality_n"] += 1
    if major:
        s["major_bouts"] += 1

    if won is None:
        return

    s["recent_results"].append(1 if won else 0)
    if won:
        s["wins"] += 1
        s["win_streak"] = max(1, s["win_streak"] + 1)
        if major:
            s["major_wins"] += 1
        if is_finish(method):
            s["finish_wins"] += 1
        bucket = method_bucket(method)
        if bucket == "KO":
            s["ko_wins"] += 1
        elif bucket == "SUB":
            s["sub_wins"] += 1
        elif bucket == "DEC":
            s["dec_wins"] += 1
        if opp_decided >= 5 and (opp_wp >= 0.65 or opp_pre["elo"] >= 1550):
            s["quality_wins"] += 1
    else:
        s["losses"] += 1
        s["win_streak"] = 0
        if is_finish(method):
            s["finish_losses"] += 1

def build_timelines(database: Path, max_date: str):
    con = duckdb.connect(str(database), read_only=True)
    q = """
        SELECT fight_id, event_date, organization, fighter_1, fighter_2, winner,
               method_normalized, is_major_org
        FROM fights_career_longitudinal
        WHERE event_date < CAST(? AS DATE)
        ORDER BY event_date, fight_id
    """
    fights = con.execute(q, [max_date]).fetchdf()
    con.close()

    states = defaultdict(state_template)
    timelines = defaultdict(list)
    seen_names = set()

    for r in fights.itertuples(index=False):
        date = pd.Timestamp(r.event_date)
        f1, f2 = norm_name(r.fighter_1), norm_name(r.fighter_2)
        if not f1 or not f2 or f1 == f2:
            continue
        seen_names.update((f1, f2))
        s1, s2 = states[f1], states[f2]
        pre1, pre2 = dict(s1), dict(s2)
        pre1["recent_results"] = deque(s1["recent_results"], maxlen=5)
        pre2["recent_results"] = deque(s2["recent_results"], maxlen=5)

        side = result_side(r.winner, f1, f2)
        w1 = True if side == 1 else False if side == 2 else None
        w2 = True if side == 2 else False if side == 1 else None

        # Elo update uses only the pre-fight ratings and only binary W/L results.
        if side in (1, 2):
            e1 = expected(pre1["elo"], pre2["elo"])
            actual1 = 1.0 if side == 1 else 0.0
            delta = 32.0 * (actual1 - e1)
            new_elo1 = pre1["elo"] + delta
            new_elo2 = pre2["elo"] - delta
        else:
            new_elo1, new_elo2 = pre1["elo"], pre2["elo"]

        update_one(s1, pre2, w1, r.method_normalized, bool(r.is_major_org), date)
        update_one(s2, pre1, w2, r.method_normalized, bool(r.is_major_org), date)
        s1["elo"], s2["elo"] = new_elo1, new_elo2

        timelines[f1].append((date, snapshot(s1)))
        timelines[f2].append((date, snapshot(s2)))

    return timelines, len(fights), len(seen_names)

def prefight_state(timelines, fighter: str, event_date: pd.Timestamp):
    key = norm_name(fighter)
    seq = timelines.get(key)
    if not seq:
        return None
    dates = [x[0] for x in seq]
    i = bisect.bisect_left(dates, event_date) - 1
    if i < 0:
        return None
    snap = dict(seq[i][1])
    last_date = snap.pop("last_date")
    snap["days_since_last_fight"] = float((event_date - last_date).days) if last_date is not None else np.nan
    return snap

def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]

def champion():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.35, penalty="l2", solver="lbfgs", max_iter=5000)),
    ])

def challenger():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.04, max_depth=3, max_iter=250,
            min_samples_leaf=12, l2_regularization=1.0, random_state=42,
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

def walk_forward(df, features, model_factory, min_train_events=20):
    d = df.copy()
    d["_date"] = pd.to_datetime(d.event_date)
    events = d[["_date", "event_id"]].drop_duplicates().sort_values(["_date", "event_id"]).reset_index(drop=True)
    ys, ps, folds = [], [], []
    for i in range(min_train_events, len(events)):
        event_id = str(events.iloc[i].event_id)
        cutoff = events.iloc[i]._date
        train = d[d._date < cutoff]
        test = d[d.event_id.astype(str) == event_id]
        if len(train) < 50 or len(test) == 0 or train.winner_a.nunique() < 2:
            continue
        model = model_factory()
        model.fit(train[features], train.winner_a.astype(int))
        p = model.predict_proba(test[features])[:, 1]
        y = test.winner_a.astype(int).to_numpy()
        ys.extend(y.tolist()); ps.extend(p.tolist())
        m = score(y, p)
        folds.append({
            "event_date": str(test.event_date.iloc[0]),
            "event_id": event_id,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            **m,
        })
    return pd.DataFrame(folds), score(ys, ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", type=Path, required=True)
    args = ap.parse_args()

    base = pd.read_csv(BASE_MATRIX)
    base["event_date"] = pd.to_datetime(base.event_date)
    max_date = (base.event_date.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    timelines, source_fights, source_fighters = build_timelines(args.database, max_date)

    enriched = base.copy()
    global_feature_names = [
        "history_available", "career_bouts", "career_wins", "career_losses",
        "career_win_pct", "recent3_win_pct", "recent5_win_pct",
        "career_finish_pct", "ko_win_pct", "sub_win_pct", "decision_win_pct",
        "finish_loss_pct", "avg_opp_win_pct", "avg_opp_elo",
        "quality_win_count", "elo", "major_bouts", "major_win_pct",
        "win_streak", "days_since_last_fight",
    ]

    coverage_rows = []
    values = {f"{GLOBAL_PREFIX}{f}_diff": [] for f in global_feature_names}

    for r in enriched.itertuples(index=False):
        d = pd.Timestamp(r.event_date)
        a = prefight_state(timelines, r.fighter_a, d)
        b = prefight_state(timelines, r.fighter_b, d)
        coverage_rows.append({
            "event_date": d.strftime("%Y-%m-%d"),
            "fighter_a": r.fighter_a,
            "fighter_b": r.fighter_b,
            "fighter_a_matched": int(a is not None),
            "fighter_b_matched": int(b is not None),
            "both_matched": int(a is not None and b is not None),
        })
        for f in global_feature_names:
            av = np.nan if a is None else a.get(f, np.nan)
            bv = np.nan if b is None else b.get(f, np.nan)
            values[f"{GLOBAL_PREFIX}{f}_diff"].append(
                av - bv if np.isfinite(av) and np.isfinite(bv) else np.nan
            )

    for c, vals in values.items():
        enriched[c] = vals

    enriched["event_date"] = enriched.event_date.dt.strftime("%Y-%m-%d")
    enriched.to_csv(OUT_MATRIX, index=False)

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(REPORTS / "regional_history_link_coverage.csv", index=False)

    base_features = feature_columns(base)
    augmented_features = base_features + [c for c in enriched.columns if c.startswith(GLOBAL_PREFIX)]

    base_folds, base_score = walk_forward(enriched, base_features, champion)
    aug_folds, aug_score = walk_forward(enriched, augmented_features, champion)
    gb_folds, gb_score = walk_forward(enriched, augmented_features, challenger)

    base_folds.to_csv(REPORTS / "regional_candidate_baseline_folds.csv", index=False)
    aug_folds.to_csv(REPORTS / "regional_candidate_logistic_folds.csv", index=False)
    gb_folds.to_csv(REPORTS / "regional_candidate_boosting_folds.csv", index=False)

    final_log = champion()
    final_log.fit(enriched[augmented_features], enriched.winner_a.astype(int))
    final_gb = challenger()
    final_gb.fit(enriched[augmented_features], enriched.winner_a.astype(int))
    joblib.dump(final_log, MODELS / "dwcs_w_v0_4_regional_candidate_logistic.joblib")
    joblib.dump(final_gb, MODELS / "dwcs_w_v0_4_regional_candidate_boosting.joblib")

    # Coefficients correspond to original features first; missing indicators follow.
    coef = final_log.named_steps["model"].coef_[0]
    imputer = final_log.named_steps["impute"]
    names = list(augmented_features)
    if hasattr(imputer, "indicator_") and imputer.indicator_.features_.size:
        names += [f"missing__{augmented_features[i]}" for i in imputer.indicator_.features_]
    pd.DataFrame({
        "feature": names,
        "coefficient_standardized": coef,
    }).sort_values(
        "coefficient_standardized", key=lambda s: s.abs(), ascending=False
    ).to_csv(REPORTS / "regional_candidate_logistic_coefficients.csv", index=False)

    both_rate = float(coverage.both_matched.mean())
    any_rate = float(((coverage.fighter_a_matched + coverage.fighter_b_matched) > 0).mean())

    candidates = {
        "baseline_logistic": base_score,
        "regional_logistic": aug_score,
        "regional_boosting": gb_score,
    }
    # Promotion requires probability-quality improvement versus the exact same baseline folds.
    eligible = []
    for name in ("regional_logistic", "regional_boosting"):
        m = candidates[name]
        if m["brier"] < base_score["brier"] and m["log_loss"] < base_score["log_loss"]:
            eligible.append(name)
    if eligible:
        winner = min(eligible, key=lambda n: (candidates[n]["log_loss"], candidates[n]["brier"]))
        recommendation = f"PROMOTE_{winner.upper()}"
    else:
        winner = "baseline_logistic"
        recommendation = "KEEP_DWCS_W_V0_3_BASELINE"

    report = {
        "candidate_version": "DWCS-W-v0.4-regional-career",
        "source": {
            "dataset": "MMA Global Database / fights_career_longitudinal",
            "source_repository": "LeandroIber/Database-complete-mma",
            "training_use": "external career state only; DWCS rows remain the only prediction labels",
            "raw_database_committed": False,
            "source_fights_loaded_before_dwcs_cutoff": source_fights,
            "source_unique_normalized_fighters": source_fighters,
        },
        "linkage": {
            "dwcs_bouts": int(len(enriched)),
            "both_fighters_with_prior_global_history_rate": both_rate,
            "at_least_one_fighter_with_prior_global_history_rate": any_rate,
        },
        "features_added": [c for c in enriched.columns if c.startswith(GLOBAL_PREFIX)],
        "same_fold_comparison": candidates,
        "promotion_recommendation": recommendation,
        "selected_model": winner,
        "promotion_rule": "Must improve both walk-forward Brier score and log loss versus baseline on identical folds.",
        "important_limitations": [
            "Name linkage is normalized exact-name matching; unmatched aliases remain missing rather than guessed.",
            "The source database currently ends 2026-01-31, so current-season 2026 prospect records still require weekly research/current-source updates.",
            "Regional metadata supplies career/opponent-strength state, not fabricated strike/takedown statistics.",
        ],
    }
    (REPORTS / "regional_candidate_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
