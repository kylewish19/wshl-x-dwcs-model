from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from train_secondary_models import (
    METHOD3, METHOD6, ROUND4,
    method_family, duration_seconds,
    make_logistic, make_boosting, winner_model,
    multiclass_metrics, binary_metrics,
    walk_forward_secondary, candidate_features, align_proba,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_enriched_2017_2025.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"


def prepare():
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
    return df, features


def smoothed_freq(series, classes, alpha=0.5):
    vc = series.value_counts()
    vals = np.array([float(vc.get(c, 0)) + alpha for c in classes], dtype=float)
    return vals / vals.sum()


def chronological_naive(df, min_train_events=20):
    events = (
        df[["event_date_dt", "event_id"]]
        .drop_duplicates()
        .sort_values(["event_date_dt", "event_id"])
        .reset_index(drop=True)
    )
    method_y, method_p = [], []
    round_y, round_p = [], []
    markets = ["under_0_5", "under_1_5", "under_2_5", "goes_distance", "round_2_starts", "round_3_starts"]
    dstore = {m: {"y": [], "p": []} for m in markets}

    for i in range(min_train_events, len(events)):
        cutoff = events.iloc[i].event_date_dt
        eid = str(events.iloc[i].event_id)
        tr = df[df.event_date_dt < cutoff]
        te = df[df.event_id.astype(str) == eid]
        if len(tr) < 50 or len(te) == 0:
            continue

        tm = tr[tr.method6.notna()]
        em = te[te.method6.notna()]
        if len(em):
            p = smoothed_freq(tm.method6, METHOD6)
            method_y.extend(em.method6.tolist())
            method_p.extend(np.tile(p, (len(em), 1)).tolist())

        rr = tr[tr.round4.notna()]
        er = te[te.round4.notna()]
        if len(er):
            p = smoothed_freq(rr.round4, ROUND4)
            round_y.extend(er.round4.tolist())
            round_p.extend(np.tile(p, (len(er), 1)).tolist())

        for m in markets:
            p = (float(tr[m].sum()) + 0.5) / (len(tr) + 1.0)
            dstore[m]["y"].extend(te[m].astype(int).tolist())
            dstore[m]["p"].extend([p] * len(te))

    return {
        "method": multiclass_metrics(method_y, np.asarray(method_p), METHOD6),
        "duration": {m: binary_metrics(s["y"], s["p"]) for m, s in dstore.items()},
        "round": multiclass_metrics(round_y, np.asarray(round_p), ROUND4),
    }


def promote(summary, naive):
    out = {"method": {}, "duration": {}, "round": {}}

    # METHOD: primary KPI exact winner+method.
    candidates = summary["method"]["results"]
    best_name = min(candidates, key=lambda k: (candidates[k]["log_loss"], candidates[k]["brier_multiclass"]))
    best = candidates[best_name]
    n = naive["method"]
    eligible = (
        best["log_loss"] < n["log_loss"]
        and best["brier_multiclass"] < n["brier_multiclass"]
        and best["accuracy"] >= n["accuracy"]
    )
    out["method"] = {
        "best_architecture": best_name,
        "candidate": best,
        "naive": n,
        "status": "PRODUCTION_ELIGIBLE" if eligible else "SHADOW_ONLY",
        "rule": "Must beat chronological class-frequency baseline on log loss and multiclass Brier, without lower exact accuracy.",
    }

    # DURATION: market-by-market.
    for market, info in summary["duration"].items():
        n = naive["duration"][market]
        eligible_models = []
        for arch in ["logistic", "boosting"]:
            m = info[arch]
            if m["log_loss"] < n["log_loss"] and m["brier"] < n["brier"]:
                eligible_models.append(arch)
        if eligible_models:
            best_arch = min(eligible_models, key=lambda a: (info[a]["log_loss"], info[a]["brier"]))
            status = "PRODUCTION_ELIGIBLE"
        else:
            best_arch = min(["logistic", "boosting"], key=lambda a: (info[a]["log_loss"], info[a]["brier"]))
            status = "SHADOW_ONLY"
        out["duration"][market] = {
            "best_architecture": best_arch,
            "candidate": info[best_arch],
            "naive": n,
            "status": status,
            "rule": "Must beat chronological training-prevalence baseline on both log loss and Brier.",
        }

    # ROUND.
    candidates = summary["round"]["results"]
    best_name = min(candidates, key=lambda k: (candidates[k]["log_loss"], candidates[k]["brier_multiclass"]))
    best = candidates[best_name]
    n = naive["round"]
    eligible = (
        best["log_loss"] < n["log_loss"]
        and best["brier_multiclass"] < n["brier_multiclass"]
        and best["accuracy"] >= n["accuracy"]
    )
    out["round"] = {
        "best_architecture": best_name,
        "candidate": best,
        "naive": n,
        "status": "PRODUCTION_ELIGIBLE" if eligible else "SHADOW_ONLY",
        "rule": "Must beat chronological class-frequency baseline on log loss and multiclass Brier, without lower exact accuracy.",
    }
    return out


def save_logistic_coefficients(model, features, path, label_prefix):
    """Export standardized logistic coefficients for binary or multiclass fits."""
    clf = model.named_steps["model"]
    rows = []
    classes = [str(c) for c in clf.classes_]

    if clf.coef_.shape[0] == 1 and len(classes) == 2:
        # sklearn stores one coefficient vector for the positive class.
        vectors = [
            (classes[0], -clf.coef_[0]),
            (classes[1], clf.coef_[0]),
        ]
    else:
        vectors = [(cls, clf.coef_[i]) for i, cls in enumerate(classes)]

    for cls, vector in vectors:
        for feat, coef in zip(features, vector):
            rows.append({
                "model": label_prefix,
                "class": cls,
                "feature": feat,
                "coefficient_standardized": float(coef),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def fit_and_save(df, features, promotion):
    # Method direct logistic + boosting
    m = df[df.method6.notna()].copy()
    dl = make_logistic(C=0.25).fit(m[features], m.method6)
    dg = make_boosting().fit(m[features], m.method6)
    joblib.dump(dl, MODELS / "dwcs_m_direct_logistic_v0_2.joblib")
    joblib.dump(dg, MODELS / "dwcs_m_direct_boosting_v0_2.joblib")
    save_logistic_coefficients(dl, features, REPORTS / "method_direct_logistic_coefficients_v0_2.csv", "direct_method")

    # Conditional method + winner.
    Xc = m[features].copy()
    bmask = m.winner_a.astype(int).values == 0
    Xc.loc[bmask, :] = -Xc.loc[bmask, :].values
    cm = make_logistic(C=0.25).fit(Xc, m.method3)
    wm = winner_model().fit(df[features], df.winner_a.astype(int))
    joblib.dump(cm, MODELS / "dwcs_m_conditional_method_logistic_v0_2.joblib")
    joblib.dump(wm, MODELS / "dwcs_m_conditional_winner_logistic_v0_2.joblib")
    save_logistic_coefficients(cm, features, REPORTS / "method_conditional_logistic_coefficients_v0_2.csv", "conditional_method")

    # Duration: save both candidates; registry decides whether market is eligible.
    for market in promotion["duration"]:
        ml = make_logistic(C=0.35).fit(df[features], df[market].astype(int))
        mg = make_boosting().fit(df[features], df[market].astype(int))
        joblib.dump(ml, MODELS / f"dwcs_d_{market}_logistic_v0_2.joblib")
        joblib.dump(mg, MODELS / f"dwcs_d_{market}_boosting_v0_2.joblib")
        save_logistic_coefficients(
            ml, features, REPORTS / f"duration_{market}_logistic_coefficients_v0_2.csv", f"duration_{market}"
        )

    # Round
    r = df[df.round4.notna()].copy()
    rl = make_logistic(C=0.25).fit(r[features], r.round4)
    rg = make_boosting().fit(r[features], r.round4)
    joblib.dump(rl, MODELS / "dwcs_r_logistic_v0_2.joblib")
    joblib.dump(rg, MODELS / "dwcs_r_boosting_v0_2.joblib")
    save_logistic_coefficients(rl, features, REPORTS / "round_logistic_coefficients_v0_2.csv", "round")


def main():
    df, features = prepare()
    folds, summary = walk_forward_secondary(df, features, min_train_events=20)
    naive = chronological_naive(df, min_train_events=20)
    promotion = promote(summary, naive)

    coverage = {
        "rows": int(len(df)),
        "events": int(df.event_id.nunique()),
        "feature_count": int(len(features)),
        "external_feature_count": int(sum(c.startswith("ext_") for c in features)),
        "both_external_history_pct": float(((df.ext_a_available == 1) & (df.ext_b_available == 1)).mean()),
        "at_least_one_external_history_pct": float(((df.ext_a_available == 1) | (df.ext_b_available == 1)).mean()),
    }

    report = {
        "versions": {
            "method": "DWCS-M-v0.2-global-career",
            "duration": "DWCS-D-v0.2-global-career",
            "round": "DWCS-R-v0.2-global-career",
        },
        "coverage": coverage,
        "candidate_results": summary,
        "chronological_naive_baselines": naive,
        "promotion": promotion,
        "metric_audit": "Uses explicit probability-column class ordering; binary coefficient export supports sklearn one-row coef_ shape.",
        "source_policy": {
            "prediction_labels": "DWCS only",
            "external_history": "129k-fight global MMA database used only to build pre-fight context features",
            "anti_leakage": "Only fights strictly earlier than each historical DWCS event are included in snapshots",
        },
    }

    folds.to_csv(REPORTS / "secondary_models_v0_2_walk_forward_folds.csv", index=False)
    (REPORTS / "secondary_models_v0_2_summary.json").write_text(json.dumps(report, indent=2))
    fit_and_save(df, features, promotion)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
