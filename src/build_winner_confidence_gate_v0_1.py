from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"; DIAG=ROOT/"diagnostics"; MODELS=ROOT/"models"; OFFICIAL=ROOT/"official_picks"
BASE=REPORTS/"winner_v0_7_nested_validation_predictions.csv"
MOD=DIAG/"week9_v0_7_modular_rerun.csv"
LOCK=OFFICIAL/"week9_preodds_model_lock.csv"
MODULES=["resume_form","striking","grappling","durability","cardio_decision","physical_quality"]


def model():
    return Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(C=0.40,solver="lbfgs",max_iter=5000)),
    ])


def prep(d):
    x=d.copy()
    x["full_side"]=(x.p_full>=0.5).astype(int)
    x["anchor_confidence"]=np.maximum(x.p_full,1-x.p_full)
    x["full_correct"]=((x.full_side==x.winner_a.astype(int))).astype(int) if "winner_a" in x else np.nan
    x["module_support_fraction"]=0.0
    for m in MODULES:
        x[f"support_{m}"]=((x[m]>=0.5).astype(int)==x.full_side).astype(float)
        x["module_support_fraction"]+=x[f"support_{m}"]
    x["module_support_fraction"]/=len(MODULES)
    x["full_vs_module_gap"]=(x.p_full-x.module_mean).abs()
    x["extreme_x_disagreement"]=(x.anchor_confidence-0.5)*2*x.module_spread
    return x

FEATURES=[
    "anchor_confidence","module_support_fraction","module_spread","full_vs_module_gap","extreme_x_disagreement",
    *[f"support_{m}" for m in MODULES],
]


def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    return {"n":int(len(y)),"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,np.c_[1-p,p],labels=[0,1])),"auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}


def nested_eval(d,min_train=70):
    d=prep(d).sort_values(["event_date","event_id"]).copy()
    events=d[["event_date","event_id"]].drop_duplicates().reset_index(drop=True)
    rows=[]
    for _,ev in events.iterrows():
        tr=d[d.event_date<ev.event_date]; te=d[d.event_id==ev.event_id]
        if len(tr)<min_train or te.empty: continue
        m=model().fit(tr[FEATURES],tr.full_correct.astype(int))
        p=m.predict_proba(te[FEATURES])[:,1]
        for j,r in te.reset_index(drop=True).iterrows():
            rows.append({"event_date":r.event_date,"event_id":r.event_id,"full_correct":int(r.full_correct),"p_gate":float(p[j]),"anchor_confidence":float(r.anchor_confidence)})
    out=pd.DataFrame(rows)
    if out.empty: raise RuntimeError("No confidence-gate validation rows")
    return out,{"confidence_gate":metrics(out.full_correct,out.p_gate),"raw_anchor_confidence":metrics(out.full_correct,out.anchor_confidence)}


def main():
    base=pd.read_csv(BASE)
    pred,summary=nested_eval(base)
    summary["promotion_test"]={
        "lower_brier":summary["confidence_gate"]["brier"]<summary["raw_anchor_confidence"]["brier"],
        "lower_log_loss":summary["confidence_gate"]["log_loss"]<summary["raw_anchor_confidence"]["log_loss"],
    }
    summary["earns_promotion_gate"]=all(summary["promotion_test"].values())
    train=prep(base)
    gate=model().fit(train[FEATURES],train.full_correct.astype(int))

    mods=pd.read_csv(MOD); lock=pd.read_csv(LOCK)
    cur=[]
    for _,mr in mods.iterrows():
        lr=lock[lock.fight.eq(mr.fight)].iloc[0]
        fa,fb=mr.fight.split(" vs ",1)
        pa=float(lr.winner_probability) if lr.winner_pick==fa else 1-float(lr.winner_probability)
        rec={"fight":mr.fight,"p_full":pa,"module_mean":float(np.mean([mr[f"p_{m}_a"] for m in MODULES])),"module_spread":float(mr.module_spread)}
        for m in MODULES: rec[m]=float(mr[f"p_{m}_a"])
        cur.append(rec)
    c=prep(pd.DataFrame(cur)); trust=gate.predict_proba(c[FEATURES])[:,1]
    rows=[]
    for i,r in c.iterrows():
        fa,fb=r.fight.split(" vs ",1); side=fa if r.p_full>=0.5 else fb; raw=max(float(r.p_full),1-float(r.p_full)); t=float(trust[i])
        if t>=0.75: conf="HIGH"
        elif t>=0.65: conf="MEDIUM-HIGH"
        elif t>=0.56: conf="MEDIUM"
        else: conf="LOW"
        rows.append({"fight":r.fight,"winner_side":side,"raw_side_probability":raw,"confidence_gate_probability":t,"audited_confidence":conf,"module_support_fraction":float(r.module_support_fraction),"module_spread":float(r.module_spread),"full_vs_module_gap":float(r.full_vs_module_gap)})
    out=pd.DataFrame(rows); out.to_csv(DIAG/"week9_winner_confidence_gate_v0_1.csv",index=False)
    pred.to_csv(REPORTS/"winner_confidence_gate_v0_1_nested_predictions.csv",index=False)
    joblib.dump({"version":"DWCS-W-CG-v0.1","model":gate,"features":FEATURES},MODELS/"dwcs_w_confidence_gate_v0_1.joblib")
    payload={"version":"DWCS-W-CG-v0.1","built_on":"2026-10-06 pre-Week-9 event start","purpose":"Predict whether the production winner side is trustworthy; does not replace the side model.","historical_nested_validation":summary,"week9":rows}
    (REPORTS/"winner_confidence_gate_v0_1_validation.json").write_text(json.dumps(payload,indent=2))
    regp=MODELS/"model_registry.json"; reg=json.loads(regp.read_text()); reg["DWCS-W-CG-v0.1"]={"status":"promote if probability-quality gate passes","built_on":"2026-10-06","historical_nested_validation":summary,"artifact":"models/dwcs_w_confidence_gate_v0_1.joblib","week9":"diagnostics/week9_winner_confidence_gate_v0_1.csv","note":"Separate correctness/confidence layer. Production winner side remains W-v0.6 unless a future winner challenger earns promotion."}; regp.write_text(json.dumps(reg,indent=2))
    print(json.dumps({"validation":summary,"week9":rows},indent=2))

if __name__=="__main__": main()
