from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "processed" / "dwcs_historical_prefight_matrix_regional_candidate.csv"
COVERAGE = ROOT / "reports" / "regional_history_link_coverage.csv"
OUT_JSON = ROOT / "reports" / "regional_candidate_audit.json"
OUT_SEG = ROOT / "reports" / "regional_candidate_segment_scores.csv"

NON_FEATURE = {
    "event_date", "event_id", "bout_key", "fighter_a_id", "fighter_b_id",
    "fighter_a", "fighter_b", "winner_a", "winner_name", "method",
    "finish_round", "duration_minutes",
}

def score(y, p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    pred=(p>=.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y,pred)),
        "brier": float(brier_score_loss(y,p)),
        "log_loss": float(log_loss(y,np.c_[1-p,p],labels=[0,1])),
        "auc": float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
    }

def model(add_indicator=False):
    return Pipeline([
        ("impute",SimpleImputer(strategy="median",add_indicator=add_indicator)),
        ("scale",StandardScaler()),
        ("model",LogisticRegression(C=.35,solver="lbfgs",max_iter=5000)),
    ])

def walk_predictions(df, features, add_indicator=False, min_train_events=20):
    d=df.copy()
    d["_date"]=pd.to_datetime(d.event_date)
    events=d[["_date","event_id"]].drop_duplicates().sort_values(["_date","event_id"]).reset_index(drop=True)
    out=[]
    for i in range(min_train_events,len(events)):
        eid=str(events.iloc[i].event_id); cutoff=events.iloc[i]._date
        train=d[d._date<cutoff]; test=d[d.event_id.astype(str)==eid]
        if len(train)<50 or len(test)==0 or train.winner_a.nunique()<2: continue
        m=model(add_indicator)
        m.fit(train[features],train.winner_a.astype(int))
        p=m.predict_proba(test[features])[:,1]
        for idx,y,prob in zip(test.index,test.winner_a.astype(int),p):
            out.append({"row_index":int(idx),"event_id":eid,"y":int(y),"p":float(prob)})
    return pd.DataFrame(out)

def main():
    df=pd.read_csv(MATRIX)
    cov=pd.read_csv(COVERAGE,keep_default_na=False)
    if len(df)!=len(cov):
        raise ValueError("coverage rows do not align with matrix rows")

    numeric=[c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    base=[c for c in numeric if not c.startswith("global_")]
    regional=[c for c in numeric if c.startswith("global_")]
    conservative=[c for c in numeric if c!="global_history_available_diff"]

    # Strict no-future-source audit.
    target=pd.to_datetime(cov.event_date)
    violations=0
    for col in ("fighter_a_last_source_date","fighter_b_last_source_date"):
        parsed=pd.to_datetime(cov[col],errors="coerce")
        violations += int((parsed.notna() & (parsed>=target)).sum())

    p_base=walk_predictions(df,base,False)
    p_cons=walk_predictions(df,conservative,False)

    rng=np.random.default_rng(20260916)
    shuffled=df.copy()
    order=rng.permutation(len(shuffled))
    shuffled.loc[:,regional]=shuffled.loc[order,regional].to_numpy()
    p_shuffle=walk_predictions(shuffled,conservative,False)

    base_score=score(p_base.y,p_base.p)
    cons_score=score(p_cons.y,p_cons.p)
    shuffle_score=score(p_shuffle.y,p_shuffle.p)

    merged=p_cons.merge(cov.reset_index().rename(columns={"index":"row_index"}),on="row_index",how="left")
    merged["segment"]=np.select(
        [merged.both_matched.eq(1),(merged.fighter_a_matched+merged.fighter_b_matched).eq(1)],
        ["both_matched","one_matched"],
        default="neither_matched",
    )
    seg=[]
    for name,g in merged.groupby("segment"):
        if len(g):
            seg.append({"segment":name,**score(g.y,g.p)})
    pd.DataFrame(seg).to_csv(OUT_SEG,index=False)

    audit_pass=(
        violations==0
        and cons_score["brier"] < base_score["brier"]
        and cons_score["log_loss"] < base_score["log_loss"]
        and cons_score["log_loss"] + 0.03 < shuffle_score["log_loss"]
        and float(cov.both_matched.mean()) >= 0.60
    )

    report={
        "candidate":"DWCS-W-v0.4-regional-career",
        "date_leakage_violations":violations,
        "both_matched_rate":float(cov.both_matched.mean()),
        "at_least_one_matched_rate":float(((cov.fighter_a_matched+cov.fighter_b_matched)>0).mean()),
        "baseline_same_fold":base_score,
        "regional_conservative_no_availability_feature_no_missing_indicators":cons_score,
        "shuffled_regional_control":shuffle_score,
        "audit_pass":bool(audit_pass),
        "promotion_guidance":"Eligible for promotion as logistic champion" if audit_pass else "Do not promote; investigate",
        "notes":[
            "Conservative audit removes explicit history-availability differential and disables imputer missingness indicators.",
            "Shuffle control permutes all regional features across fights with a fixed seed; it should lose the regional signal.",
            "Every linked source state must end strictly before the DWCS target event date.",
        ],
    }
    OUT_JSON.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
