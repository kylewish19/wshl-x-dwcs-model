from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/"models"/"dwcs_w_v0_4_promoted_logistic.joblib"
REGISTRY=ROOT/"models"/"model_registry.json"

META_COLUMNS={
    "event_date","event_id","fighter_a","fighter_b","both_histories_established",
    "fighter_a_snapshot_date","fighter_b_snapshot_date",
    "fighter_a_provenance","fighter_b_provenance",
}

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    return np.log(p/(1-p))

def calibrate_confidence(p,slope,intercept):
    return float(sigmoid(slope*logit([p])[0]+intercept))

def fair_american(p):
    p=float(p)
    return round(-100*p/(1-p)) if p>=0.5 else round(100*(1-p)/p)

def sha256_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    ap.add_argument("--allow-fallback",action="store_true",
                    help="Score rows failing the both-histories gate but label them FALLBACK_ONLY.")
    args=ap.parse_args()

    df=pd.read_csv(args.features)
    prohibited=[c for c in df.columns if "odds" in c.casefold() or "sportsbook" in c.casefold()]
    if prohibited:
        raise ValueError(f"Odds/sportsbook columns prohibited before model lock: {prohibited}")

    if "both_histories_established" not in df:
        raise ValueError("coverage gate column both_histories_established is required")
    failed=df.both_histories_established.astype(int).ne(1)
    if failed.any() and not args.allow_fallback:
        bad=df.loc[failed,["fighter_a","fighter_b"]].to_dict("records")
        raise ValueError(f"v0.4 coverage gate failed for {bad}; use v0.3+tape fallback or refresh both histories.")

    model=joblib.load(MODEL)
    features=list(model.feature_names_in_)
    missing=[c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"feature file missing trained columns: {missing}")

    raw_a=model.predict_proba(df[features])[:,1]
    reg=json.loads(REGISTRY.read_text())
    cal=reg["DWCS-CAL-v0.2"]
    slope=float(cal["slope"]); intercept=float(cal["intercept"])

    rows=[]
    for i,r in df.iterrows():
        p_a=float(raw_a[i])
        if p_a>=0.5:
            pick=str(r.fighter_a); raw_conf=p_a; direction="A"
        else:
            pick=str(r.fighter_b); raw_conf=1-p_a; direction="B"
        calibrated=calibrate_confidence(raw_conf,slope,intercept)
        rows.append({
            "event_date":r.event_date,
            "event_id":r.event_id,
            "fighter_a":r.fighter_a,
            "fighter_b":r.fighter_b,
            "winner_pick":pick,
            "winner_side":direction,
            "winner_probability_raw":raw_conf,
            "winner_probability_calibrated":calibrated,
            "model_fair_american":fair_american(calibrated),
            "coverage_status":"PRIMARY_V0_4" if int(r.both_histories_established)==1 else "FALLBACK_ONLY",
        })

    out=pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)

    receipt={
        "model_version":"DWCS-W-v0.4",
        "production_artifact":str(MODEL.relative_to(ROOT)),
        "model_sha256":sha256_file(MODEL),
        "calibration_version":"DWCS-CAL-v0.2",
        "calibration_slope":slope,
        "calibration_intercept":intercept,
        "odds_used":False,
        "coverage_gate":"both fighter histories required for PRIMARY_V0_4",
        "feature_rows":int(len(df)),
        "trained_feature_count":len(features),
        "input_file":str(args.features),
        "output_file":str(args.out),
    }
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(receipt,indent=2))
    print(out.to_string(index=False))
    print(json.dumps(receipt,indent=2))

if __name__=="__main__":
    main()
