from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WINNER_FEATURES_PATH = ROOT / "data" / "current" / "week7_model_features.csv"
SECONDARY_FEATURES_PATH = ROOT / "data" / "current" / "week7_secondary_model_features.csv"
OUT_DIR = ROOT / "official_picks"
REPORT_DIR = ROOT / "reports"

WINNER_MODEL = ROOT / "models" / "dwcs_w_v0_4_promoted_logistic.joblib"
WINNER_BASELINE_MODEL = ROOT / "models" / "dwcs_w_logistic_v0_3.joblib"
WINNER_CHALLENGER_MODEL = ROOT / "models" / "dwcs_w_v0_4_regional_candidate_boosting.joblib"
METHOD_MODEL = ROOT / "models" / "dwcs_m_conditional_method_logistic_v0_2.joblib"
DIRECT_METHOD_MODEL = ROOT / "models" / "dwcs_m_direct_logistic_v0_2.joblib"
ROUND_MODEL = ROOT / "models" / "dwcs_r_boosting_v0_2.joblib"

DURATION_MODELS = {
    "under_0_5": ROOT / "models" / "dwcs_d_under_0_5_boosting_v0_2.joblib",
    "under_1_5": ROOT / "models" / "dwcs_d_under_1_5_boosting_v0_2.joblib",
    "under_2_5": ROOT / "models" / "dwcs_d_under_2_5_boosting_v0_2.joblib",
    "goes_distance": ROOT / "models" / "dwcs_d_goes_distance_boosting_v0_2.joblib",
    "round_2_starts": ROOT / "models" / "dwcs_d_round_2_starts_boosting_v0_2.joblib",
    "round_3_starts": ROOT / "models" / "dwcs_d_round_3_starts_boosting_v0_2.joblib",
}

METHOD_ORDER = ["KO_TKO", "SUB", "DEC"]
JOINT_ORDER = ["A_KO_TKO", "A_SUB", "A_DEC", "B_KO_TKO", "B_SUB", "B_DEC"]
ROUND_ORDER = ["R1", "R2", "R3", "DEC"]


def aligned(model, x, order):
    raw = model.predict_proba(x)
    classes = [str(c) for c in model.classes_]
    out = np.zeros((len(x), len(order)), dtype=float)
    for j, c in enumerate(order):
        if c in classes:
            out[:, j] = raw[:, classes.index(c)]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def prob_one(model, x):
    classes = list(model.classes_)
    return model.predict_proba(x)[:, classes.index(1)]


def fair_american(p):
    p = float(np.clip(p, 1e-6, 1-1e-6))
    return int(round(-100*p/(1-p))) if p >= 0.5 else int(round(100*(1-p)/p))


def main():
    wdf = pd.read_csv(WINNER_FEATURES_PATH)
    sdf = pd.read_csv(SECONDARY_FEATURES_PATH)

    keys = ["event_date","event_id","fighter_a","fighter_b"]
    if not wdf[keys].equals(sdf[keys]):
        raise SystemExit("Winner and secondary feature files are not row-aligned.")
    if not (wdf["both_histories_established"].astype(int) == 1).all():
        raise SystemExit("Week 7 contains a row failing the v0.4 both-histories production gate.")

    winner = joblib.load(WINNER_MODEL)
    winner_baseline = joblib.load(WINNER_BASELINE_MODEL)
    winner_challenger = joblib.load(WINNER_CHALLENGER_MODEL)
    method = joblib.load(METHOD_MODEL)
    direct = joblib.load(DIRECT_METHOD_MODEL)
    round_model = joblib.load(ROUND_MODEL)
    duration = {k: joblib.load(v) for k, v in DURATION_MODELS.items()}

    wf = list(winner.feature_names_in_)
    mf = list(method.feature_names_in_)
    for label, frame, features in [("winner",wdf,wf),("secondary",sdf,mf)]:
        missing=[c for c in features if c not in frame.columns]
        if missing:
            raise SystemExit(f"{label} feature file missing trained columns: {missing}")

    p_a_win = winner.predict_proba(wdf[wf])[:, list(winner.classes_).index(1)]
    # The exploratory boosting challenger was trained with an explicit A-minus-B
    # history-availability flag. All Week 7 rows have both histories established,
    # so the correct current value is 1-1 = 0 rather than a missing field.
    if "global_history_available_diff" not in wdf.columns:
        wdf["global_history_available_diff"] = 0.0
    bf = list(winner_baseline.feature_names_in_)
    cf = list(winner_challenger.feature_names_in_)
    p_a_baseline = winner_baseline.predict_proba(wdf[bf])[:, list(winner_baseline.classes_).index(1)]
    p_a_challenger = winner_challenger.predict_proba(wdf[cf])[:, list(winner_challenger.classes_).index(1)]

    xa = sdf[mf].copy()
    xb = -sdf[mf].copy()
    pa_method = aligned(method, xa, METHOD_ORDER)
    pb_method = aligned(method, xb, METHOD_ORDER)

    joint = np.zeros((len(sdf), 6), dtype=float)
    joint[:, 0:3] = p_a_win[:, None] * pa_method
    joint[:, 3:6] = (1.0 - p_a_win)[:, None] * pb_method
    joint /= joint.sum(axis=1, keepdims=True)

    direct_joint = aligned(direct, sdf[list(direct.feature_names_in_)], JOINT_ORDER)

    dur_probs = {}
    for market, model in duration.items():
        dur_probs[market] = prob_one(model, sdf[list(model.feature_names_in_)])

    round_probs = aligned(round_model, sdf[list(round_model.feature_names_in_)], ROUND_ORDER)

    rows, sims = [], []
    seeds = np.random.SeedSequence(20260922).spawn(len(sdf))

    for i, r in wdf.iterrows():
        f_a, f_b = str(r.fighter_a), str(r.fighter_b)
        p_a = float(p_a_win[i])
        pick = f_a if p_a >= 0.5 else f_b
        p_pick = max(p_a, 1-p_a)

        jp = joint[i]
        top_joint_idx = int(np.argmax(jp))
        top_joint_cls = JOINT_ORDER[top_joint_idx]
        side, method_name = top_joint_cls.split("_", 1)
        method_fighter = f_a if side == "A" else f_b

        direct_idx = int(np.argmax(direct_joint[i]))
        direct_cls = JOINT_ORDER[direct_idx]

        rp = round_probs[i]
        top_round = ROUND_ORDER[int(np.argmax(rp))]

        rng = np.random.default_rng(seeds[i])
        method_draws = rng.choice(len(JOINT_ORDER), size=10000, p=jp)
        round_draws = rng.choice(len(ROUND_ORDER), size=10000, p=rp)
        sim_joint = {JOINT_ORDER[j]: int((method_draws == j).sum()) for j in range(6)}
        sim_round = {ROUND_ORDER[j]: int((round_draws == j).sum()) for j in range(4)}
        sim_duration = {
            market: int(rng.binomial(10000, float(dur_probs[market][i])))
            for market in DURATION_MODELS
        }

        rows.append({
            "fight": f"{f_a} vs {f_b}",
            "winner_pick": pick,
            "winner_probability": p_pick,
            "winner_fair_american": fair_american(p_pick),
            "winner_v0_3_a_probability": float(p_a_baseline[i]),
            "winner_v0_4_boosting_a_probability": float(p_a_challenger[i]),
            "winner_all_models_same_side": ((p_a>=0.5)==(p_a_baseline[i]>=0.5)==(p_a_challenger[i]>=0.5)),
            "top_winner_method": f"{method_fighter} by {method_name}",
            "top_winner_method_probability": float(jp[top_joint_idx]),
            "direct_method_top_class": direct_cls,
            "direct_method_top_probability": float(direct_joint[i][direct_idx]),
            "method_model_agreement": direct_cls == top_joint_cls,
            "round_top": top_round,
            "round_top_probability_shadow": float(rp.max()),
            "under_0_5_probability_shadow": float(dur_probs["under_0_5"][i]),
            "under_1_5_probability_shadow": float(dur_probs["under_1_5"][i]),
            "under_2_5_probability_shadow": float(dur_probs["under_2_5"][i]),
            "goes_distance_probability_shadow": float(dur_probs["goes_distance"][i]),
            "round_2_starts_probability_shadow": float(dur_probs["round_2_starts"][i]),
            "round_3_starts_probability_shadow": float(dur_probs["round_3_starts"][i]),
            "coverage_status": "PRIMARY_V0_4",
            "odds_used": False,
        })

        sims.append({
            "fight": f"{f_a} vs {f_b}",
            "joint_outcome_counts_10000": sim_joint,
            "round_counts_10000": sim_round,
            "duration_yes_counts_10000": sim_duration,
        })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "week7_preodds_model_lock.csv", index=False)

    detail=[]
    for i,r in wdf.iterrows():
        detail.append({
            "fight": f"{r.fighter_a} vs {r.fighter_b}",
            "winner": {str(r.fighter_a):float(p_a_win[i]), str(r.fighter_b):float(1-p_a_win[i])},
            "winner_diagnostics": {
                "v0_3_baseline_a_probability": float(p_a_baseline[i]),
                "v0_4_boosting_challenger_a_probability": float(p_a_challenger[i])
            },
            "joint_winner_method": {JOINT_ORDER[j]:float(joint[i,j]) for j in range(6)},
            "direct_six_class_diagnostic": {JOINT_ORDER[j]:float(direct_joint[i,j]) for j in range(6)},
            "round_shadow": {ROUND_ORDER[j]:float(round_probs[i,j]) for j in range(4)},
            "duration_shadow": {m:float(dur_probs[m][i]) for m in DURATION_MODELS},
            "monte_carlo": sims[i],
        })

    payload={
        "event":"DWCS Season 10 Week 7",
        "event_date":"2026-09-22",
        "lock_type":"PRE_ODDS",
        "odds_used":False,
        "models":{
            "winner":"DWCS-W-v0.4 promoted logistic",
            "method":"DWCS-M-v0.2 conditional winner x method logistic",
            "duration":"DWCS-D-v0.2 shadow boosting",
            "round":"DWCS-R-v0.2 shadow boosting"
        },
        "simulation_runs_per_fight":10000,
        "secondary_feature_schema":"Rebuilt from same ext_* historical feature engineering used to train M/D/R v0.2, with known 2026 post-cutoff fights appended without fabricated duration/round data.",
        "note":"Duration and round remain shadow-only because historical probability metrics did not beat chronological naive baselines.",
        "fights":detail
    }
    (REPORT_DIR/"week7_preodds_model_lock.json").write_text(json.dumps(payload,indent=2))
    print(out.to_string(index=False))


if __name__=="__main__":
    main()
