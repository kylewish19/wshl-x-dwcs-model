from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_2017_2025.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

METHOD3 = ["KO_TKO", "SUB", "DEC"]
METHOD6 = ["A_KO_TKO", "A_SUB", "A_DEC", "B_KO_TKO", "B_SUB", "B_DEC"]
ROUND4 = ["R1", "R2", "R3", "DEC"]


def method_family(x):
    s = str(x).upper()
    if "KO" in s or "TKO" in s:
        return "KO_TKO"
    if "SUB" in s:
        return "SUB"
    if "DEC" in s:
        return "DEC"
    return None


def duration_seconds(v):
    """ESPN mirror stores elapsed time like 6.23 for 6:23, not decimal minutes."""
    if pd.isna(v):
        return np.nan
    x = float(v)
    minutes = int(np.floor(x + 1e-9))
    seconds = int(round((x - minutes) * 100))
    if seconds >= 60:
        # defensive fallback if source ever switches to true decimal minutes
        return int(round(x * 60))
    return minutes * 60 + seconds


def make_logistic(C=0.35):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=5000,
        )),
    ])


def make_boosting():
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


def align_proba(model, X, classes):
    raw = model.predict_proba(X)
    got = [str(x) for x in model.classes_]
    out = np.full((len(X), len(classes)), 1e-12, dtype=float)
    for j, c in enumerate(classes):
        if c in got:
            out[:, j] = raw[:, got.index(c)]
    out = out / out.sum(axis=1, keepdims=True)
    return out


def multiclass_metrics(y, p, classes):
    """Metrics with probability columns interpreted in the explicit class order.

    sklearn.log_loss sorts string labels lexicographically, which can silently
    mismatch a custom probability-column order. Index the observed class
    directly so the metric always matches align_proba() and our saved class list.
    """
    y = np.asarray(y).astype(str)
    p = np.asarray(p, dtype=float)
    idx = np.array([classes.index(v) for v in y], dtype=int)
    onehot = np.eye(len(classes))[idx]
    pred = np.array(classes)[np.argmax(p, axis=1)]
    eps = 1e-12
    ll = -np.mean(np.log(np.clip(p[np.arange(len(y)), idx], eps, 1.0)))
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "log_loss": float(ll),
        "brier_multiclass": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
    }


def binary_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    eps = 1e-12
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y*np.log(np.clip(p, eps, 1-eps)) + (1-y)*np.log(np.clip(1-p, eps, 1-eps)))),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return out


def winner_model():
    return make_logistic(C=0.35)


def candidate_features(X, side):
    """All production winner features are A-minus-B differences.
    For a B-candidate method row, invert them so positive always means candidate advantage.
    """
    z = X.copy()
    if side == "B":
        z = -z
    return z


def chronological_events(df):
    e = (
        df[["event_date_dt", "event_id"]]
        .drop_duplicates()
        .sort_values(["event_date_dt", "event_id"])
        .reset_index(drop=True)
    )
    return e


def walk_forward_secondary(df, features, min_train_events=20):
    events = chronological_events(df)

    direct_y, direct_log_p, direct_gb_p = [], [], []
    cond_y, cond_p = [], []
    round_y, round_log_p, round_gb_p = [], [], []

    markets = ["under_0_5", "under_1_5", "under_2_5", "goes_distance", "round_2_starts", "round_3_starts"]
    duration_store = {
        m: {"y": [], "log": [], "gb": []}
        for m in markets
    }

    fold_rows = []

    for i in range(min_train_events, len(events)):
        cutoff = events.iloc[i].event_date_dt
        eid = str(events.iloc[i].event_id)
        train = df[df.event_date_dt < cutoff].copy()
        test = df[df.event_id.astype(str) == eid].copy()

        if len(train) < 50 or len(test) == 0:
            continue

        Xtr, Xte = train[features], test[features]

        # ----- winner model used only by conditional winner+method architecture -----
        wm = winner_model()
        wm.fit(Xtr, train.winner_a.astype(int))
        p_a_win = wm.predict_proba(Xte)[:, 1]

        # ----- method: direct six-class -----
        trm = train[train.method6.notna()].copy()
        tem = test[test.method6.notna()].copy()

        if len(tem):
            direct_log = make_logistic(C=0.25)
            direct_gb = make_boosting()
            direct_log.fit(trm[features], trm.method6)
            direct_gb.fit(trm[features], trm.method6)
            pdlog = align_proba(direct_log, tem[features], METHOD6)
            pdgb = align_proba(direct_gb, tem[features], METHOD6)

            direct_y.extend(tem.method6.tolist())
            direct_log_p.extend(pdlog.tolist())
            direct_gb_p.extend(pdgb.tolist())

            # ----- method: conditional P(winner) x P(method | candidate winner) -----
            ctr = train[train.method3.notna()].copy()
            Xc = ctr[features].copy()
            bmask = ctr.winner_a.astype(int).values == 0
            Xc.loc[bmask, :] = -Xc.loc[bmask, :].values
            cm = make_logistic(C=0.25)
            cm.fit(Xc, ctr.method3)

            idx = tem.index
            pwin = pd.Series(p_a_win, index=test.index).loc[idx].to_numpy()
            pa_m = align_proba(cm, candidate_features(tem[features], "A"), METHOD3)
            pb_m = align_proba(cm, candidate_features(tem[features], "B"), METHOD3)

            joint = np.zeros((len(tem), 6), dtype=float)
            joint[:, 0:3] = pwin[:, None] * pa_m
            joint[:, 3:6] = (1.0 - pwin)[:, None] * pb_m
            joint = joint / joint.sum(axis=1, keepdims=True)

            cond_y.extend(tem.method6.tolist())
            cond_p.extend(joint.tolist())

        # ----- duration / props -----
        for market in markets:
            ytr = train[market].astype(int)
            yte = test[market].astype(int)
            if ytr.nunique() < 2:
                continue
            ml = make_logistic(C=0.35)
            mg = make_boosting()
            ml.fit(Xtr, ytr)
            mg.fit(Xtr, ytr)
            pl = ml.predict_proba(Xte)[:, list(ml.classes_).index(1)]
            pg = mg.predict_proba(Xte)[:, list(mg.classes_).index(1)]
            duration_store[market]["y"].extend(yte.tolist())
            duration_store[market]["log"].extend(pl.tolist())
            duration_store[market]["gb"].extend(pg.tolist())

        # ----- round / decision -----
        rr = train[train.round4.notna()].copy()
        rt = test[test.round4.notna()].copy()
        if len(rt):
            rl = make_logistic(C=0.25)
            rg = make_boosting()
            rl.fit(rr[features], rr.round4)
            rg.fit(rr[features], rr.round4)
            prl = align_proba(rl, rt[features], ROUND4)
            prg = align_proba(rg, rt[features], ROUND4)
            round_y.extend(rt.round4.tolist())
            round_log_p.extend(prl.tolist())
            round_gb_p.extend(prg.tolist())

        fold_rows.append({
            "event_date": str(test.event_date.iloc[0]),
            "event_id": eid,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        })

    method_results = {
        "direct_logistic": multiclass_metrics(direct_y, np.asarray(direct_log_p), METHOD6),
        "direct_boosting": multiclass_metrics(direct_y, np.asarray(direct_gb_p), METHOD6),
        "conditional_winner_x_method_logistic": multiclass_metrics(cond_y, np.asarray(cond_p), METHOD6),
    }
    method_best = min(
        method_results,
        key=lambda k: (method_results[k]["log_loss"], method_results[k]["brier_multiclass"])
    )

    duration_results = {}
    for market, s in duration_store.items():
        lm = binary_metrics(s["y"], s["log"])
        gm = binary_metrics(s["y"], s["gb"])
        winner = "logistic" if (lm["log_loss"], lm["brier"]) <= (gm["log_loss"], gm["brier"]) else "boosting"
        duration_results[market] = {
            "logistic": lm,
            "boosting": gm,
            "promoted_architecture": winner,
        }

    round_results = {
        "logistic": multiclass_metrics(round_y, np.asarray(round_log_p), ROUND4),
        "boosting": multiclass_metrics(round_y, np.asarray(round_gb_p), ROUND4),
    }
    round_best = min(
        round_results,
        key=lambda k: (round_results[k]["log_loss"], round_results[k]["brier_multiclass"])
    )

    return pd.DataFrame(fold_rows), {
        "method": {
            "classes": METHOD6,
            "results": method_results,
            "promoted_architecture": method_best,
        },
        "duration": duration_results,
        "round": {
            "classes": ROUND4,
            "results": round_results,
            "promoted_architecture": round_best,
        },
    }


def fit_final_models(df, features, summary):
    # Method direct models
    m = df[df.method6.notna()].copy()
    direct_log = make_logistic(C=0.25).fit(m[features], m.method6)
    direct_gb = make_boosting().fit(m[features], m.method6)
    joblib.dump(direct_log, MODELS / "dwcs_m_direct_logistic_v0_1.joblib")
    joblib.dump(direct_gb, MODELS / "dwcs_m_direct_boosting_v0_1.joblib")

    # Conditional method model plus winner model
    Xc = m[features].copy()
    bmask = m.winner_a.astype(int).values == 0
    Xc.loc[bmask, :] = -Xc.loc[bmask, :].values
    cond = make_logistic(C=0.25).fit(Xc, m.method3)
    win = winner_model().fit(df[features], df.winner_a.astype(int))
    joblib.dump(cond, MODELS / "dwcs_m_conditional_method_logistic_v0_1.joblib")
    joblib.dump(win, MODELS / "dwcs_m_conditional_winner_logistic_v0_1.joblib")

    # Duration models, save promoted architecture for each market
    for market, info in summary["duration"].items():
        arch = info["promoted_architecture"]
        model = make_logistic(C=0.35) if arch == "logistic" else make_boosting()
        model.fit(df[features], df[market].astype(int))
        joblib.dump(model, MODELS / f"dwcs_d_{market}_{arch}_v0_1.joblib")

    # Round models
    r = df[df.round4.notna()].copy()
    rl = make_logistic(C=0.25).fit(r[features], r.round4)
    rg = make_boosting().fit(r[features], r.round4)
    joblib.dump(rl, MODELS / "dwcs_r_logistic_v0_1.joblib")
    joblib.dump(rg, MODELS / "dwcs_r_boosting_v0_1.joblib")


def main():
    df = pd.read_csv(MATRIX)
    df["event_date_dt"] = pd.to_datetime(df.event_date)
    features = [c for c in df.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(df[c])]

    df["method3"] = df.method.map(method_family)
    df["method6"] = np.where(
        df.method3.notna(),
        np.where(df.winner_a.astype(int).eq(1), "A_", "B_") + df.method3.fillna(""),
        None,
    )

    df["duration_seconds"] = df.duration_minutes.map(duration_seconds)
    df = df[df.duration_seconds.notna()].copy()

    df["under_0_5"] = (df.duration_seconds < 150).astype(int)
    df["under_1_5"] = (df.duration_seconds < 450).astype(int)
    df["under_2_5"] = (df.duration_seconds < 750).astype(int)
    df["goes_distance"] = df.method3.eq("DEC").astype(int)
    df["round_2_starts"] = (df.finish_round.astype(int) >= 2).astype(int)
    df["round_3_starts"] = (df.finish_round.astype(int) >= 3).astype(int)
    df["round4"] = np.where(
        df.method3.eq("DEC"),
        "DEC",
        "R" + df.finish_round.astype(int).astype(str),
    )
    df.loc[~df.round4.isin(ROUND4), "round4"] = None

    folds, summary = walk_forward_secondary(df, features, min_train_events=20)

    summary["model_versions"] = {
        "method": "DWCS-M-v0.1-fighter-specific",
        "duration": "DWCS-D-v0.1-fighter-specific",
        "round": "DWCS-R-v0.1-fighter-specific",
    }
    summary["data"] = {
        "rows": int(len(df)),
        "events": int(df.event_id.nunique()),
        "date_min": str(df.event_date.min()),
        "date_max": str(df.event_date.max()),
        "feature_count": int(len(features)),
        "method_eligible_rows": int(df.method6.notna().sum()),
        "round_eligible_rows": int(df.round4.notna().sum()),
    }
    summary["notes"] = [
        "All validation is chronological event-by-event walk-forward.",
        "No sportsbook odds are model features.",
        "Method primary KPI is exact six-outcome winner+method probability quality.",
        "Conditional method architecture multiplies P(winner) by P(method | candidate winner).",
        "Duration thresholds use converted elapsed seconds because ESPN source encodes 6.23 as 6:23.",
        "Tape-derived Week 6 lesson features remain prospective-only and are not retroactively invented.",
    ]

    folds.to_csv(REPORTS / "secondary_models_walk_forward_folds.csv", index=False)
    (REPORTS / "secondary_models_summary.json").write_text(json.dumps(summary, indent=2))
    fit_final_models(df, features, summary)

    # Compact comparison tables
    mr = []
    for name, m in summary["method"]["results"].items():
        mr.append({"architecture": name, **m})
    pd.DataFrame(mr).to_csv(REPORTS / "method_model_comparison.csv", index=False)

    dr = []
    for market, info in summary["duration"].items():
        for arch in ["logistic", "boosting"]:
            dr.append({"market": market, "architecture": arch, **info[arch],
                       "promoted": arch == info["promoted_architecture"]})
    pd.DataFrame(dr).to_csv(REPORTS / "duration_model_comparison.csv", index=False)

    rr = []
    for arch, m in summary["round"]["results"].items():
        rr.append({"architecture": arch, **m,
                   "promoted": arch == summary["round"]["promoted_architecture"]})
    pd.DataFrame(rr).to_csv(REPORTS / "round_model_comparison.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
