from __future__ import annotations

# Week 8 pre-odds execution trigger.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from learn_from_week7 import METHOD3, METHOD_FEATURES, BOUNDARIES, method_vec, current_record_method_state

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "data/current"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OFFICIAL = ROOT / "official_picks"

WINNER_MODEL = MODELS / "dwcs_w_v0_5_week7_logistic.joblib"
WINNER_BASELINE = MODELS / "dwcs_w_logistic_v0_3.joblib"
WINNER_CHALLENGER = MODELS / "dwcs_w_v0_4_regional_candidate_boosting.joblib"
METHOD_MODEL = MODELS / "dwcs_m_v0_3_candidate_method_logistic.joblib"

WINNER_FEATURES = CURRENT / "week8_model_features.csv"
SECONDARY_FEATURES = CURRENT / "week8_secondary_model_features.csv"
RECORDS = CURRENT / "week8_current_records.csv"
SNAPS = CURRENT / "week8_fighter_snapshots.csv"

JOINT_ORDER = ["A_KO_TKO","A_SUB","A_DEC","B_KO_TKO","B_SUB","B_DEC"]
ROUND_ORDER = ["R1","R2","R3","DEC"]


def aligned(model, x, order):
    raw = model.predict_proba(x)
    classes = [str(c) for c in model.classes_]
    out = np.zeros((len(x), len(order)), dtype=float)
    for j,c in enumerate(order):
        if c in classes:
            out[:,j] = raw[:,classes.index(c)]
    return out / np.clip(out.sum(axis=1,keepdims=True),1e-12,None)


def load_hazards():
    return [joblib.load(MODELS/f"dwcs_d_hazard_to_{end}s_v0_3.joblib") for end in BOUNDARIES]


def hazard_distribution(models, X):
    surv=np.ones(len(X),dtype=float)
    S={}
    for end,m in zip(BOUNDARIES,models):
        h=m.predict_proba(X)[:,list(m.classes_).index(1)]
        surv=surv*(1-h)
        S[end]=surv.copy()
    interval=np.c_[
        1-S[150],
        S[150]-S[300],
        S[300]-S[450],
        S[450]-S[600],
        S[600]-S[750],
        S[750]-S[900],
        S[900],
    ]
    interval=np.clip(interval,0,1)
    interval=interval/interval.sum(axis=1,keepdims=True)
    return {
        "under_0_5":1-S[150],
        "under_1_5":1-S[450],
        "under_2_5":1-S[750],
        "goes_distance":S[900],
        "round_2_starts":S[300],
        "round_3_starts":S[600],
        "round4":np.c_[1-S[300],S[300]-S[600],S[600]-S[900],S[900]],
        "interval7":interval,
    }


def career_method_counts(rec):
    return {
        "KO_TKO":int(rec.ko_wins),
        "SUB":int(rec.sub_wins),
        "DEC":int(rec.dec_wins),
    }


def route_plausibility(rec, method_name):
    counts=career_method_counts(rec)
    wins=max(1,int(rec.wins))
    smoothed=(counts[method_name]+1)/(wins+3)
    return float(smoothed), counts[method_name]


def main():
    wdf=pd.read_csv(WINNER_FEATURES)
    sdf=pd.read_csv(SECONDARY_FEATURES)
    records=pd.read_csv(RECORDS).set_index("fighter_name")
    snaps=pd.read_csv(SNAPS).set_index("fighter_name")

    keys=["event_date","event_id","fighter_a","fighter_b"]
    if not wdf[keys].equals(sdf[keys]):
        raise SystemExit("Week 8 winner and secondary feature rows are not aligned.")
    if not (wdf.both_histories_established.astype(int)==1).all():
        raise SystemExit("Week 8 has a row failing the both-histories gate.")

    winner=joblib.load(WINNER_MODEL)
    baseline=joblib.load(WINNER_BASELINE)
    challenger=joblib.load(WINNER_CHALLENGER)
    method=joblib.load(METHOD_MODEL)
    hazards=load_hazards()

    wf=list(winner.feature_names_in_)
    missing=[c for c in wf if c not in wdf.columns]
    if missing: raise SystemExit(f"Missing winner features: {missing}")
    p_a=winner.predict_proba(wdf[wf])[:,list(winner.classes_).index(1)]

    if "global_history_available_diff" not in wdf.columns:
        wdf["global_history_available_diff"]=0.0
    bf=list(baseline.feature_names_in_)
    cf=list(challenger.feature_names_in_)
    p_base=baseline.predict_proba(wdf[bf])[:,list(baseline.classes_).index(1)]
    p_chal=challenger.predict_proba(wdf[cf])[:,list(challenger.classes_).index(1)]

    hf=list(hazards[0].feature_names_in_)
    miss=[c for c in hf if c not in sdf.columns]
    if miss: raise SystemExit(f"Missing hazard features: {miss}")
    dur=hazard_distribution(hazards,sdf[hf])

    out_rows=[]; detail=[]
    seeds=np.random.SeedSequence(20260929).spawn(len(wdf))

    for i,r in wdf.iterrows():
        fa,fb=str(r.fighter_a),str(r.fighter_b)
        ra,rb=records.loc[fa],records.loc[fb]
        sa,sb=snaps.loc[fa],snaps.loc[fb]
        ast=current_record_method_state(ra,sa)
        bst=current_record_method_state(rb,sb)
        va=pd.DataFrame([method_vec(ast,bst)])[METHOD_FEATURES]
        vb=pd.DataFrame([method_vec(bst,ast)])[METHOD_FEATURES]
        pa_m=aligned(method,va,METHOD3)[0]
        pb_m=aligned(method,vb,METHOD3)[0]

        joint=np.r_[p_a[i]*pa_m,(1-p_a[i])*pb_m]
        joint=joint/joint.sum()
        pick_a=bool(p_a[i]>=0.5)
        pick=fa if pick_a else fb
        pick_prob=float(max(p_a[i],1-p_a[i]))
        pdiag=[float(p_a[i]),float(p_base[i]),float(p_chal[i])]
        sides=[x>=0.5 for x in pdiag]
        agreement=(sides[0]==sides[1]==sides[2])
        median_a=float(np.median(pdiag))
        disagreement=float(max(pdiag)-min(pdiag))

        side_slice=joint[:3] if pick_a else joint[3:]
        cond=pa_m if pick_a else pb_m
        top_idx=int(np.argmax(side_slice))
        method_name=METHOD3[top_idx]
        method_joint=float(side_slice[top_idx])
        method_cond=float(cond[top_idx])
        rec=ra if pick_a else rb
        plaus,count=route_plausibility(rec,method_name)

        # Qualification gate: method needs a stable winner side, strong conditional
        # route, non-trivial joint probability, and plausible career evidence.
        method_qualified=bool(
            pick_prob>=0.60 and method_cond>=0.48 and method_joint>=0.30
            and plaus>=0.12 and (count>0 or method_cond>=0.65)
            and disagreement<=0.30
        )

        rp=dur["round4"][i]
        top_round=ROUND_ORDER[int(np.argmax(rp))]

        # M-primary coherent Monte Carlo: winner+method from promoted M; finish
        # timing comes from D-v0.3 conditional on a finish. DEC forces 15:00.
        rng=np.random.default_rng(seeds[i])
        md=rng.choice(6,size=10000,p=joint)
        intervals=dur["interval7"][i]
        finish_int=intervals[:6]/np.clip(intervals[:6].sum(),1e-12,None)
        interval_draw=np.full(10000,6,dtype=int)
        finmask=np.array([JOINT_ORDER[j].endswith("DEC")==False for j in md])
        interval_draw[finmask]=rng.choice(6,size=int(finmask.sum()),p=finish_int)

        sim_joint={JOINT_ORDER[j]:int((md==j).sum()) for j in range(6)}
        sim_round={
            "R1":int((interval_draw<=1).sum()),
            "R2":int(((interval_draw>=2)&(interval_draw<=3)).sum()),
            "R3":int(((interval_draw>=4)&(interval_draw<=5)).sum()),
            "DEC":int((interval_draw==6).sum()),
        }
        sim_duration={
            "under_0_5":int((interval_draw==0).sum()),
            "under_1_5":int((interval_draw<=2).sum()),
            "under_2_5":int((interval_draw<=4).sum()),
            "goes_distance":int((interval_draw==6).sum()),
            "round_2_starts":int((interval_draw>=2).sum()),
            "round_3_starts":int((interval_draw>=4).sum()),
        }

        row={
            "fight":f"{fa} vs {fb}",
            "winner_pick":pick,
            "winner_probability":pick_prob,
            "winner_models_agree":agreement,
            "winner_model_disagreement_range":disagreement,
            "winner_three_model_median_a":median_a,
            "method_forecast":f"{pick} by {method_name}",
            "method_joint_probability":method_joint,
            "method_conditional_probability_given_win":method_cond,
            "method_career_smoothed_plausibility":plaus,
            "method_career_win_count":count,
            "method_status":"QUALIFIED_PREODDS" if method_qualified else "FORECAST_ONLY",
            "round_top_shadow":top_round,
            "round_top_probability_shadow":float(rp.max()),
            "under_0_5_probability_shadow":float(dur["under_0_5"][i]),
            "under_1_5_probability_shadow":float(dur["under_1_5"][i]),
            "under_2_5_probability_shadow":float(dur["under_2_5"][i]),
            "goes_distance_probability_shadow":float(dur["goes_distance"][i]),
            "round_2_starts_probability_shadow":float(dur["round_2_starts"][i]),
            "round_3_starts_probability_shadow":float(dur["round_3_starts"][i]),
            "simulation_runs":10000,
            "odds_used":False,
        }
        out_rows.append(row)
        detail.append({
            **row,
            "winner_probabilities":{
                fa:float(p_a[i]),fb:float(1-p_a[i]),
                "v0_3_baseline_a":float(p_base[i]),
                "v0_4_boosting_challenger_a":float(p_chal[i]),
            },
            "conditional_method_if_a_wins":dict(zip(METHOD3,map(float,pa_m))),
            "conditional_method_if_b_wins":dict(zip(METHOD3,map(float,pb_m))),
            "joint_winner_method":dict(zip(JOINT_ORDER,map(float,joint))),
            "duration_shadow":{
                "under_0_5":float(dur["under_0_5"][i]),
                "under_1_5":float(dur["under_1_5"][i]),
                "under_2_5":float(dur["under_2_5"][i]),
                "goes_distance":float(dur["goes_distance"][i]),
                "round_2_starts":float(dur["round_2_starts"][i]),
                "round_3_starts":float(dur["round_3_starts"][i]),
            },
            "round_shadow":dict(zip(ROUND_ORDER,map(float,rp))),
            "monte_carlo":{
                "joint_outcome_counts_10000":sim_joint,
                "round_counts_10000":sim_round,
                "duration_counts_10000":sim_duration,
                "note":"Winner+method uses promoted M-v0.3; finish timing uses D-v0.3 conditional on a finish. D/R standalone probabilities remain shadow."
            }
        })

    out=pd.DataFrame(out_rows)
    OFFICIAL.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    out.to_csv(OFFICIAL/"week8_preodds_model_lock.csv",index=False)
    payload={
        "event":"DWCS Season 10 Week 8",
        "event_date":"2026-09-29",
        "lock_type":"PRE_ODDS",
        "odds_used":False,
        "models":{
            "winner":"DWCS-W-v0.5 current-season retrain",
            "method":"DWCS-M-v0.3 candidate-access promoted",
            "duration":"DWCS-D-v0.3 coherent hazard SHADOW",
            "round":"DWCS-R-v0.3 hazard-derived SHADOW",
        },
        "simulation_runs_per_fight":10000,
        "fights":detail,
    }
    (REPORTS/"week8_preodds_model_lock.json").write_text(json.dumps(payload,indent=2))
    print(out.to_string(index=False))


if __name__=="__main__":
    main()
