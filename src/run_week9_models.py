from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from learn_from_week7 import METHOD3, BOUNDARIES, method_vec, current_record_method_state
from learn_from_week8 import (
    HIER_FEATURES, PAIR_FEATURES, add_hier_features, add_pair_features,
    candidate_frame, hier_conditional_probs,
)

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data/current"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OFFICIAL = ROOT / "official_picks"

WINNER_MODEL = MODELS / "dwcs_w_v0_6_week8_logistic.joblib"
WINNER_BOOST = MODELS / "dwcs_w_v0_6_boosting_shadow.joblib"
WINNER_BASELINE = MODELS / "dwcs_w_logistic_v0_3.joblib"
METHOD_FINISH = MODELS / "dwcs_m_v0_4_finish_gate_logistic.joblib"
METHOD_ROUTE = MODELS / "dwcs_m_v0_4_finish_route_logistic.joblib"

WINNER_FEATURES = CURRENT / "week9_model_features.csv"
RECORDS = CURRENT / "week9_current_records.csv"
SNAPS = CURRENT / "week9_fighter_snapshots.csv"
TAPE = CURRENT / "week9_tape_features.csv"

JOINT_ORDER = ["A_KO_TKO","A_SUB","A_DEC","B_KO_TKO","B_SUB","B_DEC"]
ROUND_ORDER = ["R1","R2","R3","DEC"]


def load_hazards():
    return [joblib.load(MODELS / f"dwcs_d_pair_hazard_to_{end}s_v0_4.joblib") for end in BOUNDARIES]


def hazard_distribution(models, X):
    surv = np.ones(len(X), dtype=float)
    S = {}
    for end, model in zip(BOUNDARIES, models):
        h = model.predict_proba(X)[:, list(model.classes_).index(1)]
        surv *= (1.0 - h)
        S[end] = surv.copy()
    interval = np.c_[
        1-S[150], S[150]-S[300], S[300]-S[450],
        S[450]-S[600], S[600]-S[750], S[750]-S[900], S[900]
    ]
    interval = np.clip(interval, 0.0, 1.0)
    interval /= np.clip(interval.sum(axis=1, keepdims=True), 1e-12, None)
    return {
        "under_0_5":1-S[150],
        "under_1_5":1-S[450],
        "under_2_5":1-S[750],
        "goes_distance":S[900],
        "round_2_starts":S[300],
        "round_3_starts":S[600],
        "round4":np.c_[1-S[300], S[300]-S[600], S[600]-S[900], S[900]],
        "interval7":interval,
    }


def method_pair_row(ra, rb, sa, sb):
    ast = current_record_method_state(ra, sa)
    bst = current_record_method_state(rb, sb)
    va = method_vec(ast, bst)
    vb = method_vec(bst, ast)
    row = {f"a__{k}":v for k,v in va.items()}
    row.update({f"b__{k}":v for k,v in vb.items()})
    df = add_hier_features(pd.DataFrame([row]))
    return df


def route_plausibility(rec, method_name):
    counts = {
        "KO_TKO": int(rec.ko_wins),
        "SUB": int(rec.sub_wins),
        "DEC": int(rec.dec_wins),
    }
    wins = max(1, int(rec.wins))
    return float((counts[method_name] + 1) / (wins + 3)), counts[method_name]


def main():
    wdf = pd.read_csv(WINNER_FEATURES)
    records = pd.read_csv(RECORDS).set_index("fighter_name")
    snaps = pd.read_csv(SNAPS).set_index("fighter_name")
    tape = pd.read_csv(TAPE).set_index("fighter_name")

    winner = joblib.load(WINNER_MODEL)
    boost = joblib.load(WINNER_BOOST)
    baseline = joblib.load(WINNER_BASELINE)
    finish_model = joblib.load(METHOD_FINISH)
    route_model = joblib.load(METHOD_ROUTE)
    hazards = load_hazards()

    wf = list(winner.feature_names_in_)
    missing = [c for c in wf if c not in wdf.columns]
    if missing:
        raise SystemExit(f"Missing Week 9 winner features: {missing}")
    p_prod_a = winner.predict_proba(wdf[wf])[:, list(winner.classes_).index(1)]

    boostf = list(boost.feature_names_in_)
    p_boost_a = boost.predict_proba(wdf[boostf])[:, list(boost.classes_).index(1)]

    if "global_history_available_diff" not in wdf.columns:
        wdf["global_history_available_diff"] = 0.0
    basef = list(baseline.feature_names_in_)
    p_base_a = baseline.predict_proba(wdf[basef])[:, list(baseline.classes_).index(1)]

    out_rows = []
    detail = []
    seeds = np.random.SeedSequence(20261006).spawn(len(wdf))

    for i, r in wdf.iterrows():
        fa, fb = str(r.fighter_a), str(r.fighter_b)
        ra, rb = records.loc[fa], records.loc[fb]
        sa, sb = snaps.loc[fa], snaps.loc[fb]
        tpa, tpb = tape.loc[fa], tape.loc[fb]

        mrow = method_pair_row(ra, rb, sa, sb)
        xa = candidate_frame(mrow, "a__", HIER_FEATURES)
        xb = candidate_frame(mrow, "b__", HIER_FEATURES)
        pa_m = hier_conditional_probs(finish_model, route_model, xa)[0]
        pb_m = hier_conditional_probs(finish_model, route_model, xb)[0]

        pair = add_pair_features(mrow)
        dur = hazard_distribution(hazards, pair[PAIR_FEATURES])

        coverage = int(r.both_histories_established) == 1
        prod_a = float(p_prod_a[i])
        boost_a = float(p_boost_a[i])
        base_a = float(p_base_a[i])
        effective_a = prod_a if coverage else base_a
        pick_a = effective_a >= 0.5
        pick = fa if pick_a else fb
        pick_prob = max(effective_a, 1-effective_a)

        joint = np.r_[prod_a*pa_m, (1-prod_a)*pb_m]
        joint /= joint.sum()
        side_slice = joint[:3] if pick_a else joint[3:]
        cond = pa_m if pick_a else pb_m
        top_idx = int(np.argmax(side_slice))
        method_name = METHOD3[top_idx]
        method_joint = float(side_slice[top_idx])
        method_cond = float(cond[top_idx])
        rec = ra if pick_a else rb
        plaus, method_count = route_plausibility(rec, method_name)

        diagnostics = [prod_a, boost_a, base_a]
        sides = [p >= 0.5 for p in diagnostics]
        agree = sides[0] == sides[1] == sides[2]
        disagreement = max(diagnostics) - min(diagnostics)

        tape_pick = tpa if pick_a else tpb
        special_tko_access = max(
            float(tape_pick.get("grappling_generated_tko_access", 0) or 0),
            float(tape_pick.get("slam_ground_damage_tko_access", 0) or 0),
            float(tape_pick.get("early_finish_route_strength", 0) or 0),
        )
        route_supported = method_count > 0 or method_cond >= 0.65 or (method_name == "KO_TKO" and special_tko_access >= 3)
        method_qualified = bool(
            coverage and pick_prob >= 0.60 and method_cond >= 0.48 and method_joint >= 0.28
            and route_supported and disagreement <= 0.30
        )

        rp = dur["round4"][0]
        top_round = ROUND_ORDER[int(np.argmax(rp))]

        rng = np.random.default_rng(seeds[i])
        md = rng.choice(6, size=10000, p=joint)
        intervals = dur["interval7"][0]
        finish_intervals = intervals[:6] / np.clip(intervals[:6].sum(), 1e-12, None)
        interval_draw = np.full(10000, 6, dtype=int)
        finish_mask = np.array([not JOINT_ORDER[j].endswith("DEC") for j in md])
        interval_draw[finish_mask] = rng.choice(6, size=int(finish_mask.sum()), p=finish_intervals)

        sim_joint = {JOINT_ORDER[j]:int((md==j).sum()) for j in range(6)}
        sim_round = {
            "R1":int((interval_draw<=1).sum()),
            "R2":int(((interval_draw>=2)&(interval_draw<=3)).sum()),
            "R3":int(((interval_draw>=4)&(interval_draw<=5)).sum()),
            "DEC":int((interval_draw==6).sum()),
        }

        row = {
            "fight":f"{fa} vs {fb}",
            "winner_pick":pick,
            "winner_probability":float(pick_prob),
            "coverage_status":"PRIMARY_V0_6" if coverage else "V0_3_FALLBACK",
            "winner_models_agree":bool(agree),
            "winner_model_disagreement_range":float(disagreement),
            "method_forecast":f"{pick} by {method_name}",
            "method_joint_probability":method_joint,
            "method_conditional_probability_given_win":method_cond,
            "method_status":"QUALIFIED_PREODDS" if method_qualified else "FORECAST_ONLY",
            "under_0_5_probability":float(dur["under_0_5"][0]),
            "under_1_5_probability":float(dur["under_1_5"][0]),
            "under_2_5_probability":float(dur["under_2_5"][0]),
            "goes_distance_probability":float(dur["goes_distance"][0]),
            "round_2_starts_probability":float(dur["round_2_starts"][0]),
            "round_3_starts_probability":float(dur["round_3_starts"][0]),
            "round_top":top_round,
            "round_top_probability":float(rp.max()),
            "simulation_runs":10000,
            "odds_used":False,
        }
        out_rows.append(row)
        detail.append({
            **row,
            "winner_probabilities":{
                fa:prod_a,
                fb:1-prod_a,
                "v0_6_boosting_shadow_a":boost_a,
                "v0_3_fallback_a":base_a,
            },
            "conditional_method_if_a_wins":dict(zip(METHOD3, map(float, pa_m))),
            "conditional_method_if_b_wins":dict(zip(METHOD3, map(float, pb_m))),
            "joint_winner_method":dict(zip(JOINT_ORDER, map(float, joint))),
            "duration":{
                "under_0_5":float(dur["under_0_5"][0]),
                "under_1_5":float(dur["under_1_5"][0]),
                "under_2_5":float(dur["under_2_5"][0]),
                "goes_distance":float(dur["goes_distance"][0]),
                "round_2_starts":float(dur["round_2_starts"][0]),
                "round_3_starts":float(dur["round_3_starts"][0]),
            },
            "round":dict(zip(ROUND_ORDER, map(float, rp))),
            "method_route_gate":{
                "career_smoothed_plausibility":plaus,
                "career_method_win_count":method_count,
                "special_tko_access_score":special_tko_access,
                "route_supported":bool(route_supported),
            },
            "prospective_lesson_features":{
                "pick_fighter_evidence_class":str(tape_pick.evidence_class),
                "elite_amateur_pedigree":None if pd.isna(tape_pick.elite_amateur_pedigree) else float(tape_pick.elite_amateur_pedigree),
                "pre_ufc_championship_experience":None if pd.isna(tape_pick.pre_ufc_championship_experience) else float(tape_pick.pre_ufc_championship_experience),
                "five_round_experience":None if pd.isna(tape_pick.five_round_experience) else float(tape_pick.five_round_experience),
                "post_high_output_round_freshness":None if pd.isna(tape_pick.post_high_output_round_freshness) else float(tape_pick.post_high_output_round_freshness),
            },
            "monte_carlo":{
                "joint_outcome_counts_10000":sim_joint,
                "round_counts_10000":sim_round,
                "note":"Joint winner+method uses W-v0.6 and M-v0.4. Finish timing uses D-v0.4 conditional on a finish; standalone D/R probabilities are separately reported as production outputs."
            }
        })

    OFFICIAL.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(out_rows).to_csv(OFFICIAL / "week9_preodds_model_lock.csv", index=False)
    payload = {
        "event":"DWCS Season 10 Week 9",
        "event_date":"2026-10-06",
        "lock_type":"PRE_ODDS",
        "odds_used":False,
        "models":{
            "winner":"DWCS-W-v0.6",
            "method":"DWCS-M-v0.4 hierarchical finish-route",
            "duration":"DWCS-D-v0.4 pair-symmetric coherent hazard",
            "round":"DWCS-R-v0.4 hazard-derived",
        },
        "simulation_runs_per_fight":10000,
        "fights":detail,
    }
    (REPORTS / "week9_preodds_model_lock.json").write_text(json.dumps(payload, indent=2))
    print(pd.DataFrame(out_rows).to_string(index=False))


if __name__ == "__main__":
    main()
