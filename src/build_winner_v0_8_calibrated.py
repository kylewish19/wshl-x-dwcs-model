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
REPORTS = ROOT / "reports"
DIAG = ROOT / "diagnostics"
MODELS = ROOT / "models"
OFFICIAL = ROOT / "official_picks"

VAL_PRED = REPORTS / "winner_v0_7_nested_validation_predictions.csv"
MOD_RERUN = DIAG / "week9_v0_7_modular_rerun.csv"
W9LOCK = OFFICIAL / "week9_preodds_model_lock.csv"

MODULES = ["resume_form", "striking", "grappling", "durability", "cardio_decision", "physical_quality"]


def make_model(C=0.35):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, solver="lbfgs", max_iter=5000)),
    ])


def metrics(y, p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pred=(p>=0.5).astype(int)
    return {
        "n":int(len(y)),
        "accuracy":float(accuracy_score(y,pred)),
        "brier":float(brier_score_loss(y,p)),
        "log_loss":float(log_loss(y,np.c_[1-p,p],labels=[0,1])),
        "auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
    }


def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-5,1-1e-5)
    return np.log(p/(1-p))


def add_fusion_features(df):
    d=df.copy()
    d["logit_full"]=logit(d.p_full)
    d["logit_module_mean"]=logit(d.module_mean)
    d["full_vs_module_gap"]=(d.p_full-d.module_mean).abs()
    d["full_confidence"]=(d.p_full-0.5).abs()*2
    d["disagreement_at_extreme"]=d["logit_full"]*d["module_spread"]
    d["gap_at_extreme"]=d["logit_full"]*d["full_vs_module_gap"]
    return d


FUSION_FEATURES=[
    "logit_full","logit_module_mean","module_spread","full_vs_module_gap","full_confidence",
    "disagreement_at_extreme","gap_at_extreme",
    *MODULES,
]


def nested_eval(base, min_train_rows=70):
    d=add_fusion_features(base).sort_values(["event_date","event_id"]).copy()
    events=d[["event_date","event_id"]].drop_duplicates().reset_index(drop=True)
    rows=[]
    for _,ev in events.iterrows():
        tr=d[d.event_date < ev.event_date]
        te=d[d.event_id == ev.event_id]
        if len(tr)<min_train_rows or te.empty:
            continue
        m=make_model().fit(tr[FUSION_FEATURES],tr.winner_a.astype(int))
        p=m.predict_proba(te[FUSION_FEATURES])[:,1]
        for j,r in te.reset_index(drop=True).iterrows():
            rows.append({"event_date":r.event_date,"event_id":r.event_id,"winner_a":int(r.winner_a),"p_v0_8":float(p[j]),"p_full":float(r.p_full)})
    pred=pd.DataFrame(rows)
    if pred.empty: raise RuntimeError("No v0.8 evaluation rows")
    return pred, {
        "v0_8_evidence_calibrated":metrics(pred.winner_a,pred.p_v0_8),
        "full_feature_anchor_same_rows":metrics(pred.winner_a,pred.p_full),
    }


def main():
    base=pd.read_csv(VAL_PRED)
    pred,summary=nested_eval(base)
    summary["promotion_test"]={
        "lower_brier":summary["v0_8_evidence_calibrated"]["brier"] < summary["full_feature_anchor_same_rows"]["brier"],
        "lower_log_loss":summary["v0_8_evidence_calibrated"]["log_loss"] < summary["full_feature_anchor_same_rows"]["log_loss"],
        "accuracy_not_lower":summary["v0_8_evidence_calibrated"]["accuracy"] >= summary["full_feature_anchor_same_rows"]["accuracy"],
    }
    summary["earns_historical_promotion_gate"]=all(summary["promotion_test"].values())

    final_train=add_fusion_features(base)
    fusion=make_model().fit(final_train[FUSION_FEATURES],final_train.winner_a.astype(int))

    mods=pd.read_csv(MOD_RERUN)
    lock=pd.read_csv(W9LOCK)
    rows=[]
    for _,mr in mods.iterrows():
        fight=mr.fight
        lr=lock[lock.fight.eq(fight)].iloc[0]
        fa,fb=fight.split(" vs ",1)
        if lr.winner_pick==fa:
            p_full=float(lr.winner_probability)
        else:
            p_full=1-float(lr.winner_probability)
        rec={"fight":fight,"p_full":p_full}
        for m in MODULES:
            rec[m]=float(mr[f"p_{m}_a"])
        rec["module_mean"]=float(np.mean([rec[m] for m in MODULES]))
        rec["module_spread"]=float(np.max([rec[m] for m in MODULES])-np.min([rec[m] for m in MODULES]))
        rows.append(rec)
    cur=add_fusion_features(pd.DataFrame(rows))
    p=fusion.predict_proba(cur[FUSION_FEATURES])[:,1]

    out=[]
    for i,r in cur.iterrows():
        fa,fb=r.fight.split(" vs ",1)
        pa=float(p[i]); pick=fa if pa>=0.5 else fb; pp=max(pa,1-pa)
        direct=str(mods.loc[mods.fight.eq(r.fight),"direct_skill_stat_coverage"].iloc[0])
        spread=float(r.module_spread)
        warning=None
        if pp>=0.80 and (direct=="LOW" or spread>0.30): warning="EXTREME_PROBABILITY_REQUIRES_CAUTION"
        if pp>=0.70 and spread<=0.20 and direct!="LOW": conf="HIGH"
        elif pp>=0.62 and spread<=0.30: conf="MEDIUM-HIGH" if direct!="LOW" else "MEDIUM"
        elif pp>=0.56: conf="MEDIUM" if spread<=0.35 else "LOW"
        else: conf="LOW"
        out.append({
            "fight":r.fight,"v0_8_pick":pick,"v0_8_probability":pp,"p_fighter_a":pa,
            "anchor_probability_a":float(r.p_full),"module_mean_a":float(r.module_mean),"module_spread":spread,
            "direct_skill_stat_coverage":direct,"audited_confidence":conf,"warning":warning,
        })
    outdf=pd.DataFrame(out)
    outdf.to_csv(DIAG/"week9_v0_8_evidence_calibrated_rerun.csv",index=False)
    pred.to_csv(REPORTS/"winner_v0_8_nested_validation_predictions.csv",index=False)
    joblib.dump({"version":"DWCS-W-v0.8-evidence-calibrated-challenger","model":fusion,"features":FUSION_FEATURES},MODELS/"dwcs_w_v0_8_evidence_calibrated_challenger.joblib")

    payload={
        "version":"DWCS-W-v0.8-evidence-calibrated-challenger",
        "built_on":"2026-10-06 pre-Week-9 event start",
        "status":"DEVELOPMENT CHALLENGER — original Week 9 lock unchanged",
        "architecture":"Existing full winner signal anchored by a learned evidence-calibration layer using modular skill/resume probabilities, module spread, and full-vs-module disagreement. No manual probability haircut.",
        "historical_nested_validation":summary,
        "week9_rerun":out,
    }
    (REPORTS/"winner_v0_8_evidence_calibrated_validation.json").write_text(json.dumps(payload,indent=2))

    registry_path=MODELS/"model_registry.json"
    reg=json.loads(registry_path.read_text())
    reg["DWCS-W-v0.8-evidence-calibrated-challenger"]={
        "status":"development challenger; original Week 9 lock preserved",
        "built_on":"2026-10-06",
        "historical_nested_validation":summary,
        "artifact":"models/dwcs_w_v0_8_evidence_calibrated_challenger.joblib",
        "rerun":"diagnostics/week9_v0_8_evidence_calibrated_rerun.csv",
        "note":"Learns whether module disagreement should attenuate or reinforce the full-model probability; no hand-edited probability caps.",
    }
    registry_path.write_text(json.dumps(reg,indent=2))
    print(json.dumps({"validation":summary,"week9":out},indent=2))

if __name__=="__main__": main()
