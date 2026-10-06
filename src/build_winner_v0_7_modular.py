from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CURRENT = DATA / "current"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics"

HIST = PROCESSED / "dwcs_historical_prefight_matrix_regional_candidate.csv"
W7F = CURRENT / "week7_model_features.csv"
W7R = CURRENT / "week7_results.csv"
W8F = CURRENT / "week8_model_features.csv"
W8R = CURRENT / "week8_results.csv"
W9F = CURRENT / "week9_model_features.csv"
W9S = CURRENT / "week9_fighter_snapshots.csv"
W9T = CURRENT / "week9_tape_features.csv"

NON_FEATURE = {
    "event_date", "event_id", "bout_key", "fighter_a_id", "fighter_b_id",
    "fighter_a", "fighter_b", "winner_a", "winner_name", "method",
    "finish_round", "duration_minutes", "round", "time", "source", "split",
}

MODULES = {
    "resume_form": [
        "global_career_bouts_diff", "global_career_wins_diff", "global_career_losses_diff",
        "global_career_win_pct_diff", "global_recent3_win_pct_diff", "global_recent5_win_pct_diff",
        "global_win_streak_diff", "global_elo_diff", "global_avg_opp_win_pct_diff",
        "global_avg_opp_elo_diff", "global_quality_win_count_diff", "global_major_bouts_diff",
        "global_major_win_pct_diff", "global_days_since_last_fight_diff",
    ],
    "striking": [
        "striker_style_diff", "prior_dwcs_sig_landed_pm_diff", "prior_dwcs_sig_attempted_pm_diff",
        "prior_dwcs_sig_accuracy_diff", "prior_dwcs_kd_per15_diff", "prior_dwcs_ko_win_pct_diff",
        "global_ko_win_pct_diff", "global_career_finish_pct_diff", "global_finish_loss_pct_diff",
    ],
    "grappling": [
        "grappler_style_diff", "prior_dwcs_td_landed_per15_diff", "prior_dwcs_td_attempted_per15_diff",
        "prior_dwcs_td_accuracy_diff", "prior_dwcs_sub_attempts_per15_diff", "prior_dwcs_sub_win_pct_diff",
        "global_sub_win_pct_diff", "global_finish_loss_pct_diff",
    ],
    "durability": [
        "global_finish_loss_pct_diff", "prior_dwcs_avg_duration_diff", "prior_dwcs_over25_rate_diff",
        "global_career_losses_diff", "global_decision_win_pct_diff", "global_recent5_win_pct_diff",
    ],
    "cardio_decision": [
        "prior_dwcs_avg_duration_diff", "prior_dwcs_over25_rate_diff", "prior_dwcs_dec_win_pct_diff",
        "global_decision_win_pct_diff", "global_recent5_win_pct_diff", "global_days_since_last_fight_diff",
        "global_win_streak_diff",
    ],
    "physical_quality": [
        "age_diff", "height_diff", "reach_diff", "height_missing_diff", "reach_missing_diff",
        "southpaw_diff", "switch_diff", "global_avg_opp_win_pct_diff", "global_avg_opp_elo_diff",
        "global_quality_win_count_diff", "global_major_bouts_diff", "global_major_win_pct_diff",
        "global_elo_diff",
    ],
}

TAPE_AUDIT_FIELDS = [
    "submission_access_quality", "back_exposure_created", "back_exposure_allowed",
    "durability_after_clean_damage", "failed_finish_cardio_cost", "late_round_momentum_reversal",
    "size_physicality", "decision_resilience", "elite_amateur_pedigree", "amateur_fight_sample",
    "pre_ufc_championship_experience", "five_round_experience", "grappling_generated_tko_access",
    "slam_ground_damage_tko_access", "early_finish_route_strength", "post_high_output_round_freshness",
]


def make_model(C: float = 0.35) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000)),
    ])


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.c_[1-p, p], labels=[0, 1])),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    return out


def available_features(df, requested):
    return [c for c in requested if c in df.columns]


def all_numeric_features(df):
    return [
        c for c in df.columns
        if c not in NON_FEATURE and c != "global_history_available_diff" and pd.api.types.is_numeric_dtype(df[c])
    ]


def labeled_current(feature_path: Path, result_path: Path) -> pd.DataFrame:
    f = pd.read_csv(feature_path)
    r = pd.read_csv(result_path)[["event_date", "event_id", "fighter_a", "fighter_b", "winner_a"]]
    out = f.merge(r, on=["event_date", "event_id", "fighter_a", "fighter_b"], how="inner", validate="one_to_one")
    return out


def fit_modules(train: pd.DataFrame):
    models = {}
    for name, requested in MODULES.items():
        feats = available_features(train, requested)
        m = make_model().fit(train[feats], train.winner_a.astype(int))
        models[name] = {"model": m, "features": feats}
    return models


def module_predict(models, frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name, pack in models.items():
        p = pack["model"].predict_proba(frame[pack["features"]])[:, 1]
        out[name] = p
    out["module_mean"] = out[list(MODULES)].mean(axis=1)
    out["module_spread"] = out[list(MODULES)].max(axis=1) - out[list(MODULES)].min(axis=1)
    return out


def chronological_oof(hist: pd.DataFrame, min_train_events: int = 20) -> pd.DataFrame:
    d = hist.sort_values(["event_date", "event_id"]).copy()
    events = d[["event_date", "event_id"]].drop_duplicates().reset_index(drop=True)
    full_features = all_numeric_features(d)
    rows = []
    for i in range(min_train_events, len(events)):
        ev = events.iloc[i]
        train = d[d.event_date < ev.event_date]
        test = d[d.event_id == ev.event_id]
        if train.empty or test.empty:
            continue
        mods = fit_modules(train)
        mp = module_predict(mods, test).reset_index(drop=True)
        full = make_model().fit(train[full_features], train.winner_a.astype(int))
        p_full = full.predict_proba(test[full_features])[:, 1]
        for j, rr in test.reset_index(drop=True).iterrows():
            rec = {
                "event_date": str(rr.event_date), "event_id": str(rr.event_id),
                "fighter_a": rr.fighter_a, "fighter_b": rr.fighter_b,
                "winner_a": int(rr.winner_a), "p_full": float(p_full[j]),
            }
            for c in list(MODULES) + ["module_mean", "module_spread"]:
                rec[c] = float(mp.loc[j, c])
            rows.append(rec)
    return pd.DataFrame(rows)


def meta_features():
    return list(MODULES) + ["module_mean", "module_spread"]


def nested_meta_eval(oof: pd.DataFrame, min_meta_rows: int = 60):
    oof = oof.sort_values(["event_date", "event_id"]).copy()
    events = oof[["event_date", "event_id"]].drop_duplicates().reset_index(drop=True)
    pred_rows = []
    mf = meta_features()
    for _, ev in events.iterrows():
        train = oof[oof.event_date < ev.event_date]
        test = oof[oof.event_id == ev.event_id]
        if len(train) < min_meta_rows or test.empty:
            continue
        meta = make_model(C=0.50).fit(train[mf], train.winner_a.astype(int))
        p = meta.predict_proba(test[mf])[:, 1]
        for j, rr in test.reset_index(drop=True).iterrows():
            pred_rows.append({
                "event_date": rr.event_date, "event_id": rr.event_id,
                "winner_a": int(rr.winner_a), "p_meta": float(p[j]), "p_full": float(rr.p_full),
                **{k: float(rr[k]) for k in mf},
            })
    pred = pd.DataFrame(pred_rows)
    if pred.empty:
        raise RuntimeError("No nested meta evaluation rows produced")
    summary = {
        "modular_meta": metrics(pred.winner_a, pred.p_meta),
        "full_feature_logistic_same_rows": metrics(pred.winner_a, pred.p_full),
        "module_same_rows": {name: metrics(pred.winner_a, pred[name]) for name in MODULES},
    }
    summary["promotion_test"] = {
        "lower_brier": summary["modular_meta"]["brier"] < summary["full_feature_logistic_same_rows"]["brier"],
        "lower_log_loss": summary["modular_meta"]["log_loss"] < summary["full_feature_logistic_same_rows"]["log_loss"],
        "accuracy_not_lower": summary["modular_meta"]["accuracy"] >= summary["full_feature_logistic_same_rows"]["accuracy"],
    }
    summary["earns_historical_promotion_gate"] = all(summary["promotion_test"].values())
    return pred, summary


def current_oof_row(train: pd.DataFrame, current: pd.DataFrame, source: str) -> pd.DataFrame:
    mods = fit_modules(train)
    mp = module_predict(mods, current).reset_index(drop=True)
    out = current[["event_date", "event_id", "fighter_a", "fighter_b", "winner_a"]].reset_index(drop=True).copy()
    for c in meta_features():
        out[c] = mp[c]
    out["source"] = source
    return out


def tape_audit(tape, fa, fb):
    a = tape.loc[fa]
    b = tape.loc[fb]
    diffs = {}
    for f in TAPE_AUDIT_FIELDS:
        av = a.get(f, np.nan); bv = b.get(f, np.nan)
        if pd.isna(av) or pd.isna(bv):
            diffs[f] = None
        else:
            diffs[f] = float(av - bv)
    return {
        "fighter_a_evidence": str(a.evidence_class),
        "fighter_b_evidence": str(b.evidence_class),
        "a_minus_b_dimension_diffs": diffs,
        "note": "Prospectively frozen tape fields are audit-only in v0.7; no retrospective coefficients are invented. They begin accumulating labeled evidence for future trained inclusion.",
    }


def confidence_audit(p_a, module_row, snap_a, snap_b):
    pick_p = max(float(p_a), 1.0 - float(p_a))
    probs = np.array([float(module_row[n]) for n in MODULES])
    same_side = bool(np.all(probs >= 0.5) or np.all(probs < 0.5))
    direct_a = int(float(snap_a.get("prior_dwcs_stats_available", 0) or 0) > 0)
    direct_b = int(float(snap_b.get("prior_dwcs_stats_available", 0) or 0) > 0)
    direct_coverage = "HIGH" if direct_a and direct_b else ("PARTIAL" if direct_a or direct_b else "LOW")
    spread = float(probs.max() - probs.min())
    if pick_p >= 0.75 and same_side and spread <= 0.20 and direct_coverage == "HIGH":
        label = "HIGH"
    elif pick_p >= 0.65 and same_side and spread <= 0.30:
        label = "MEDIUM-HIGH" if direct_coverage != "LOW" else "MEDIUM"
    elif pick_p >= 0.58 and spread <= 0.35:
        label = "MEDIUM"
    else:
        label = "LOW"
    warning = None
    if pick_p >= 0.80 and direct_coverage == "LOW":
        warning = "EXTREME_PROBABILITY_WITH_LOW_DIRECT_SKILL_DATA"
    return {
        "audited_confidence": label,
        "direct_skill_stat_coverage": direct_coverage,
        "all_modules_same_side": same_side,
        "module_spread": spread,
        "warning": warning,
    }


def main():
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True); DIAG.mkdir(exist_ok=True)
    hist = pd.read_csv(HIST)
    hist = hist[hist.winner_a.notna()].copy()
    hist["event_date"] = pd.to_datetime(hist.event_date).dt.strftime("%Y-%m-%d")
    hist["winner_a"] = hist.winner_a.astype(int)

    oof = chronological_oof(hist)
    nested_pred, validation = nested_meta_eval(oof)

    w7 = labeled_current(W7F, W7R)
    w8 = labeled_current(W8F, W8R)
    for d in (w7, w8):
        d["event_date"] = pd.to_datetime(d.event_date).dt.strftime("%Y-%m-%d")
        d["winner_a"] = d.winner_a.astype(int)

    w7_meta = current_oof_row(hist, w7, "week7_chronological_development")
    train_w8 = pd.concat([hist, w7], ignore_index=True, sort=False)
    w8_meta = current_oof_row(train_w8, w8, "week8_chronological_development")

    mf = meta_features()
    meta_train = pd.concat([
        oof[["event_date", "event_id", "fighter_a", "fighter_b", "winner_a"] + mf].assign(source="historical_oof"),
        w7_meta, w8_meta,
    ], ignore_index=True, sort=False)
    meta = make_model(C=0.50).fit(meta_train[mf], meta_train.winner_a.astype(int))

    final_train = pd.concat([hist, w7, w8], ignore_index=True, sort=False)
    final_modules = fit_modules(final_train)

    w9 = pd.read_csv(W9F)
    w9mp = module_predict(final_modules, w9).reset_index(drop=True)
    p_meta = meta.predict_proba(w9mp[mf])[:, 1]
    snaps = pd.read_csv(W9S).set_index("fighter_name")
    tape = pd.read_csv(W9T).set_index("fighter_name")

    rerun_rows = []
    details = []
    for i, rr in w9.reset_index(drop=True).iterrows():
        fa, fb = str(rr.fighter_a), str(rr.fighter_b)
        p_a = float(p_meta[i])
        pick_a = p_a >= 0.5
        pick = fa if pick_a else fb
        pick_p = p_a if pick_a else 1-p_a
        mods = {name: float(w9mp.loc[i, name]) for name in MODULES}
        audit = confidence_audit(p_a, w9mp.loc[i], snaps.loc[fa], snaps.loc[fb])
        rerun_rows.append({
            "event_date": rr.event_date, "event_id": rr.event_id,
            "fight": f"{fa} vs {fb}", "v0_7_pick": pick,
            "v0_7_probability": pick_p, "p_fighter_a": p_a,
            "audited_confidence": audit["audited_confidence"],
            "direct_skill_stat_coverage": audit["direct_skill_stat_coverage"],
            "module_spread": audit["module_spread"], "warning": audit["warning"],
            **{f"p_{k}_a": v for k, v in mods.items()},
        })
        details.append({
            "fight": f"{fa} vs {fb}", "fighter_a": fa, "fighter_b": fb,
            "v0_7_probability_a": p_a, "v0_7_pick": pick, "v0_7_pick_probability": pick_p,
            "module_probabilities_a": mods, "confidence_audit": audit,
            "tape_audit": tape_audit(tape, fa, fb),
        })

    rerun = pd.DataFrame(rerun_rows)
    rerun.to_csv(DIAG / "week9_v0_7_modular_rerun.csv", index=False)
    nested_pred.to_csv(REPORTS / "winner_v0_7_nested_validation_predictions.csv", index=False)

    artifact = {
        "version": "DWCS-W-v0.7-modular-challenger",
        "modules": final_modules,
        "meta_model": meta,
        "meta_features": mf,
        "module_feature_contract": MODULES,
    }
    joblib.dump(artifact, MODELS / "dwcs_w_v0_7_modular_challenger.joblib")

    payload = {
        "version": "DWCS-W-v0.7-modular-challenger",
        "built_on": "2026-10-06 pre-Week-9 event start",
        "status": "DEVELOPMENT CHALLENGER — original Week 9 lock remains unchanged",
        "design": {
            "modules": MODULES,
            "meta": "L2 logistic stacking model trained only on chronological out-of-fold module predictions plus Week 7/8 chronological development rows",
            "tape_rule": "Prospectively frozen tape concepts are displayed as audit dimensions but are not assigned invented historical weights. Future labeled weeks can train them legitimately.",
        },
        "historical_nested_validation": validation,
        "meta_training_rows": int(len(meta_train)),
        "final_training_rows": int(len(final_train)),
        "week9_rerun": details,
        "integrity": [
            "official_picks/week9_locked_card.csv is not modified",
            "no Week 9 outcomes or sportsbook odds are inputs",
            "Week 7/8 labels enter only after their frozen pre-fight feature rows",
            "Week 9 tape scores are audit-only until enough prospective labels exist to train their weights",
        ],
    }
    (REPORTS / "winner_v0_7_modular_validation.json").write_text(json.dumps(payload, indent=2))

    registry_path = MODELS / "model_registry.json"
    registry = json.loads(registry_path.read_text())
    registry["DWCS-W-v0.7-modular-challenger"] = {
        "status": "development challenger; Week 9 rerun only, original lock preserved",
        "built_on": "2026-10-06",
        "training_rows": int(len(final_train)),
        "historical_nested_validation": validation,
        "artifact": "models/dwcs_w_v0_7_modular_challenger.joblib",
        "rerun": "diagnostics/week9_v0_7_modular_rerun.csv",
        "note": "Modular real-ML architecture separates resume/form, striking, grappling, durability, cardio/decision, and physical/opponent-quality signals. Prospectively frozen tape concepts remain audit-only until trained prospectively.",
    }
    registry_path.write_text(json.dumps(registry, indent=2))

    print(json.dumps({
        "validation": validation,
        "week9": rerun[["fight", "v0_7_pick", "v0_7_probability", "audited_confidence", "direct_skill_stat_coverage", "module_spread", "warning"]].to_dict("records"),
    }, indent=2))


if __name__ == "__main__":
    main()
