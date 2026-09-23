from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from enrich_with_global_career import build_histories, norm_name, snapshot
from train_secondary_models import (
    METHOD3, METHOD6, ROUND4, method_family, duration_seconds,
    multiclass_metrics, binary_metrics, align_proba
)

ROOT = Path(__file__).resolve().parents[1]
WIN_HIST = ROOT / "data/processed/dwcs_historical_prefight_matrix_regional_candidate.csv"
SEC_HIST = ROOT / "data/processed/dwcs_historical_prefight_matrix_enriched_2017_2025.csv"
W7_WIN = ROOT / "data/current/week7_model_features.csv"
W7_SEC = ROOT / "data/current/week7_secondary_model_features.csv"
W7_RESULTS = ROOT / "data/current/week7_results.csv"
W7_RECORDS = ROOT / "data/current/week7_current_records.csv"
W7_SNAPS = ROOT / "data/current/week7_fighter_snapshots.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
PROCESSED = ROOT / "data/processed"
for p in (REPORTS, MODELS, PROCESSED):
    p.mkdir(parents=True, exist_ok=True)

NON_FEATURE = {
    "event_date","event_id","bout_key","fighter_a_id","fighter_b_id",
    "fighter_a","fighter_b","winner_a","winner_name","method",
    "finish_round","duration_minutes",
}
BOUNDARIES = [150, 300, 450, 600, 750, 900]

METHOD_FEATURES = [
    "cand_wins","cand_losses","cand_finish_win_pct","cand_ko_win_pct",
    "cand_sub_win_pct","cand_dec_win_pct","cand_recent5_finish_pct",
    "cand_r1_finish_win_pct","cand_early_finish_win_pct",
    "cand_multi_route_finisher","cand_avg_duration_sec","cand_elo",
    "opp_losses","opp_finish_loss_pct","opp_ko_loss_pct","opp_sub_loss_pct",
    "opp_avg_duration_sec","opp_elo","elo_edge","experience_edge",
    "ko_access","sub_access","decision_route","finish_conversion_vs_vulnerability",
]


def make_logistic(C=0.25):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, solver="lbfgs", max_iter=5000)),
    ])


def stable_flip(event_id, fighter_a, fighter_b):
    key = f"{event_id}|{fighter_a}|{fighter_b}".encode()
    return int(hashlib.sha256(key).hexdigest()[:2], 16) % 2 == 1


def orient_current(frame, feature_cols):
    out = frame.copy()
    for i, r in out.iterrows():
        if stable_flip(r.event_id, r.fighter_a, r.fighter_b):
            out.loc[i, feature_cols] = -out.loc[i, feature_cols].astype(float)
            out.loc[i, "winner_a"] = 1 - int(r.winner_a)
    return out


def winner_retrain():
    hist = pd.read_csv(WIN_HIST)
    cur = pd.read_csv(W7_WIN).merge(
        pd.read_csv(W7_RESULTS),
        on=["event_date","event_id","fighter_a","fighter_b"],
        how="inner",
        validate="one_to_one",
        suffixes=("","_result"),
    )
    numeric = [c for c in hist.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(hist[c])]
    features = [c for c in numeric if c != "global_history_available_diff"]
    missing = [c for c in features if c not in cur.columns]
    if missing:
        raise ValueError(f"Week 7 winner feature file missing: {missing}")
    cur = orient_current(cur, features)

    before = make_logistic(C=0.35).fit(hist[features], hist.winner_a.astype(int))
    combined = pd.concat([hist[features + ["winner_a"]], cur[features + ["winner_a"]]], ignore_index=True)
    after = make_logistic(C=0.35).fit(combined[features], combined.winner_a.astype(int))
    joblib.dump(after, MODELS / "dwcs_w_v0_5_week7_logistic.joblib")

    b = pd.Series(before.named_steps["model"].coef_[0], index=features)
    a = pd.Series(after.named_steps["model"].coef_[0], index=features)
    delta = pd.DataFrame({
        "feature": features,
        "coefficient_before": b.values,
        "coefficient_after": a.values,
        "delta": (a-b).values,
    })
    delta["abs_delta"] = delta.delta.abs()
    delta = delta.sort_values("abs_delta", ascending=False)
    delta.to_csv(REPORTS / "winner_v0_5_week7_coefficient_delta.csv", index=False)
    pd.DataFrame({"feature":features,"coefficient_standardized":a.values}).sort_values(
        "coefficient_standardized", key=lambda s:s.abs(), ascending=False
    ).to_csv(REPORTS / "winner_v0_5_week7_coefficients.csv", index=False)

    return {
        "version":"DWCS-W-v0.5-week7-retrain",
        "historical_rows":int(len(hist)),
        "new_week7_rows":int(len(cur)),
        "training_rows":int(len(combined)),
        "orientation_control":"Current rows deterministically hash-oriented before fitting to avoid card-order bias.",
        "largest_coefficient_changes":delta.head(10).to_dict("records"),
        "artifact":"models/dwcs_w_v0_5_week7_logistic.joblib",
    }


def load_global_histories():
    data_dir = Path(kagglehub.dataset_download("leandroiber/mmastats"))
    dbs = list(data_dir.rglob("*.duckdb"))
    if not dbs:
        raise FileNotFoundError("No MMA Global DuckDB found")
    con = duckdb.connect(str(dbs[0]), read_only=True)
    g = con.execute("""
        SELECT fight_id, organization, event_date, fighter_1, fighter_2, winner,
               method_normalized, round_num, time_finish_seconds,
               is_major_org, is_title_fight
        FROM fights_career_longitudinal
        WHERE event_date < DATE '2025-09-25'
        ORDER BY event_date, fight_id
    """).fetchdf()
    con.close()
    return build_histories(g)


def method_vec(cand, opp):
    cw = float(cand.get("ext_wins",0) or 0)
    cl = float(cand.get("ext_losses",0) or 0)
    ol = float(opp.get("ext_losses",0) or 0)
    cko = float(cand.get("ext_ko_win_pct",0) or 0)
    csub = float(cand.get("ext_sub_win_pct",0) or 0)
    cdec = float(cand.get("ext_dec_win_pct",0) or 0)
    cfin = float(cand.get("ext_finish_win_pct",0) or 0)
    ofl = float(opp.get("ext_finish_loss_pct",0) or 0)
    okl = float(opp.get("ext_ko_loss_pct",0) or 0)
    osl = float(opp.get("ext_sub_loss_pct",0) or 0)
    return {
        "cand_wins":cw,
        "cand_losses":cl,
        "cand_finish_win_pct":cfin,
        "cand_ko_win_pct":cko,
        "cand_sub_win_pct":csub,
        "cand_dec_win_pct":cdec,
        "cand_recent5_finish_pct":cand.get("ext_recent5_finish_pct",np.nan),
        "cand_r1_finish_win_pct":cand.get("ext_r1_finish_win_pct",np.nan),
        "cand_early_finish_win_pct":cand.get("ext_early_finish_win_pct",np.nan),
        "cand_multi_route_finisher":cand.get("ext_multi_route_finisher",np.nan),
        "cand_avg_duration_sec":cand.get("ext_avg_duration_sec",np.nan),
        "cand_elo":cand.get("ext_elo",np.nan),
        "opp_losses":ol,
        "opp_finish_loss_pct":ofl,
        "opp_ko_loss_pct":okl,
        "opp_sub_loss_pct":osl,
        "opp_avg_duration_sec":opp.get("ext_avg_duration_sec",np.nan),
        "opp_elo":opp.get("ext_elo",np.nan),
        "elo_edge":float(cand.get("ext_elo",1500) or 1500)-float(opp.get("ext_elo",1500) or 1500),
        "experience_edge":float(cand.get("ext_fights",0) or 0)-float(opp.get("ext_fights",0) or 0),
        "ko_access":cko*okl,
        "sub_access":csub*osl,
        "decision_route":cdec*(1.0-ofl),
        "finish_conversion_vs_vulnerability":cfin*ofl,
    }


def historical_method_matrix():
    base = pd.read_csv(SEC_HIST)
    histories = load_global_histories()
    rows=[]
    for r in base.itertuples(index=False):
        mf = method_family(r.method)
        if mf not in METHOD3:
            continue
        d = pd.Timestamp(r.event_date)
        sa = snapshot(histories.get(norm_name(r.fighter_a), []), d)
        sb = snapshot(histories.get(norm_name(r.fighter_b), []), d)
        va = method_vec(sa,sb)
        vb = method_vec(sb,sa)
        row={
            "event_date":str(r.event_date),"event_id":str(r.event_id),
            "fighter_a":r.fighter_a,"fighter_b":r.fighter_b,
            "winner_a":int(r.winner_a),"method3":mf,
        }
        for f in METHOD_FEATURES:
            row[f"a__{f}"]=va[f]
            row[f"b__{f}"]=vb[f]
        rows.append(row)
    out=pd.DataFrame(rows)
    out.to_csv(PROCESSED / "dwcs_method_candidate_matrix_v0_3.csv", index=False)
    return out


def current_record_method_state(rec, snap):
    wins=float(rec.wins); losses=float(rec.losses)
    return {
        "ext_wins":wins,
        "ext_losses":losses,
        "ext_finish_win_pct":(float(rec.ko_wins)+float(rec.sub_wins))/wins if wins else 0.0,
        "ext_ko_win_pct":float(rec.ko_wins)/wins if wins else 0.0,
        "ext_sub_win_pct":float(rec.sub_wins)/wins if wins else 0.0,
        "ext_dec_win_pct":float(rec.dec_wins)/wins if wins else 0.0,
        "ext_recent5_finish_pct":np.nan,
        "ext_r1_finish_win_pct":np.nan,
        "ext_early_finish_win_pct":np.nan,
        "ext_multi_route_finisher":float(float(rec.ko_wins)>0 and float(rec.sub_wins)>0),
        "ext_avg_duration_sec":np.nan,
        "ext_elo":float(snap.global_elo),
        "ext_fights":wins+losses+float(rec.nc),
        "ext_finish_loss_pct":(float(rec.ko_losses)+float(rec.sub_losses))/losses if losses else 0.0,
        "ext_ko_loss_pct":float(rec.ko_losses)/losses if losses else 0.0,
        "ext_sub_loss_pct":float(rec.sub_losses)/losses if losses else 0.0,
    }


def week7_method_rows():
    records=pd.read_csv(W7_RECORDS).set_index("fighter_name")
    snaps=pd.read_csv(W7_SNAPS).set_index("fighter_name")
    res=pd.read_csv(W7_RESULTS)
    rows=[]
    for r in res.itertuples(index=False):
        astate=current_record_method_state(records.loc[r.fighter_a],snaps.loc[r.fighter_a])
        bstate=current_record_method_state(records.loc[r.fighter_b],snaps.loc[r.fighter_b])
        va=method_vec(astate,bstate); vb=method_vec(bstate,astate)
        row={
            "event_date":r.event_date,"event_id":r.event_id,
            "fighter_a":r.fighter_a,"fighter_b":r.fighter_b,
            "winner_a":int(r.winner_a),"method3":method_family(r.method),
        }
        for f in METHOD_FEATURES:
            row[f"a__{f}"]=va[f]
            row[f"b__{f}"]=vb[f]
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_training_xy(df):
    X=[]; y=[]
    for r in df.itertuples(index=False):
        prefix="a__" if int(r.winner_a)==1 else "b__"
        X.append([getattr(r,prefix+f) for f in METHOD_FEATURES])
        y.append(r.method3)
    return pd.DataFrame(X,columns=METHOD_FEATURES), np.asarray(y)


def method_walk_forward(df, min_train_events=20):
    d=df.copy(); d["_date"]=pd.to_datetime(d.event_date)
    events=d[["_date","event_id"]].drop_duplicates().sort_values(["_date","event_id"]).reset_index(drop=True)
    ys=[]; ps=[]
    for i in range(min_train_events,len(events)):
        cutoff=events.iloc[i]._date; eid=str(events.iloc[i].event_id)
        tr=d[d._date<cutoff]; te=d[d.event_id.astype(str)==eid]
        if len(tr)<50 or len(te)==0:
            continue
        Xtr,ytr=candidate_training_xy(tr)
        if len(np.unique(ytr))<3:
            continue
        m=make_logistic(C=0.20).fit(Xtr,ytr)
        xa=pd.DataFrame([{f:getattr(r,"a__"+f) for f in METHOD_FEATURES} for r in te.itertuples(index=False)])
        xb=pd.DataFrame([{f:getattr(r,"b__"+f) for f in METHOD_FEATURES} for r in te.itertuples(index=False)])
        pa=align_proba(m,xa,METHOD3); pb=align_proba(m,xb,METHOD3)

        # Historical winner component for the same fold.
        wh=pd.read_csv(WIN_HIST)
        wh["_date"]=pd.to_datetime(wh.event_date)
        wtr=wh[wh._date<cutoff]; wte=wh[wh.event_id.astype(str)==eid]
        wfeatures=[c for c in wh.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(wh[c]) and c!="global_history_available_diff"]
        wm=make_logistic(C=0.35).fit(wtr[wfeatures],wtr.winner_a.astype(int))
        pwin=wm.predict_proba(wte[wfeatures])[:,1]
        keyprob={(str(rr.fighter_a),str(rr.fighter_b)):float(p) for rr,p in zip(wte.itertuples(index=False),pwin)}

        for j,r in enumerate(te.itertuples(index=False)):
            p_a=keyprob[(str(r.fighter_a),str(r.fighter_b))]
            joint=np.r_[p_a*pa[j],(1-p_a)*pb[j]]
            joint=joint/joint.sum()
            ys.append(("A_" if int(r.winner_a)==1 else "B_")+r.method3)
            ps.append(joint.tolist())
    return multiclass_metrics(ys,np.asarray(ps),METHOD6)


def train_method_v03():
    hist=historical_method_matrix()
    historical_metrics=method_walk_forward(hist)
    current=week7_method_rows()
    combined=pd.concat([hist,current],ignore_index=True)
    X,y=candidate_training_xy(combined)
    m=make_logistic(C=0.20).fit(X,y)
    joblib.dump(m,MODELS/"dwcs_m_v0_3_candidate_method_logistic.joblib")
    coef=m.named_steps["model"].coef_
    rows=[]
    for ci,cls in enumerate(m.named_steps["model"].classes_):
        for f,c in zip(METHOD_FEATURES,coef[ci]):
            rows.append({"class":str(cls),"feature":f,"coefficient_standardized":float(c)})
    pd.DataFrame(rows).to_csv(REPORTS/"method_v0_3_coefficients.csv",index=False)
    return {
        "version":"DWCS-M-v0.3-candidate-access",
        "historical_walk_forward":historical_metrics,
        "previous_v0_2_reference":{"accuracy":0.4357366771,"log_loss":1.5067506031,"brier_multiclass":0.6886081439},
        "week7_rows_added":int(len(current)),
        "training_rows":int(len(combined)),
        "architecture":"P(winner) x P(method|candidate winner), with candidate offense, opponent vulnerability and explicit route-access interactions.",
        "artifact":"models/dwcs_m_v0_3_candidate_method_logistic.joblib",
        "promotion_note":"Promote only if walk-forward probability quality is not worse than v0.2 and Week 8 pre-fight plausibility checks pass.",
    }


def duration_targets(df):
    d=df.copy()
    d["duration_seconds"]=d.duration_minutes.map(duration_seconds)
    return d[d.duration_seconds.notna()].copy()


def fit_hazards(train, features):
    models=[]
    for k,end in enumerate(BOUNDARIES):
        start=0 if k==0 else BOUNDARIES[k-1]
        tr=train[train.duration_seconds>=start].copy()
        y=(tr.duration_seconds<end).astype(int)
        if y.nunique()<2:
            models.append(None)
            continue
        models.append(make_logistic(C=0.20).fit(tr[features],y))
    return models


def hazard_probs(models,X):
    surv=np.ones(len(X),dtype=float)
    S={}
    for end,m in zip(BOUNDARIES,models):
        if m is None:
            h=np.zeros(len(X))
        else:
            h=m.predict_proba(X)[:,list(m.classes_).index(1)]
        surv=surv*(1-h)
        S[end]=surv.copy()
    return {
        "under_0_5":1-S[150],
        "under_1_5":1-S[450],
        "under_2_5":1-S[750],
        "goes_distance":S[900],
        "round_2_starts":S[300],
        "round_3_starts":S[600],
        "round4":np.c_[1-S[300],S[300]-S[600],S[600]-S[900],S[900]],
    }


def duration_walk_forward(hist,features,min_train_events=20):
    d=duration_targets(hist); d["_date"]=pd.to_datetime(d.event_date)
    events=d[["_date","event_id"]].drop_duplicates().sort_values(["_date","event_id"]).reset_index(drop=True)
    stores={m:{"y":[],"p":[]} for m in ["under_0_5","under_1_5","under_2_5","goes_distance","round_2_starts","round_3_starts"]}
    ry=[]; rp=[]
    for i in range(min_train_events,len(events)):
        cutoff=events.iloc[i]._date; eid=str(events.iloc[i].event_id)
        tr=d[d._date<cutoff]; te=d[d.event_id.astype(str)==eid]
        if len(tr)<50 or len(te)==0: continue
        models=fit_hazards(tr,features)
        p=hazard_probs(models,te[features])
        sec=te.duration_seconds.to_numpy()
        labels={
            "under_0_5":(sec<150).astype(int),
            "under_1_5":(sec<450).astype(int),
            "under_2_5":(sec<750).astype(int),
            "goes_distance":(sec>=900).astype(int),
            "round_2_starts":(sec>=300).astype(int),
            "round_3_starts":(sec>=600).astype(int),
        }
        for m in stores:
            stores[m]["y"].extend(labels[m].tolist())
            stores[m]["p"].extend(p[m].tolist())
        rlab=np.where(sec>=900,"DEC",np.where(sec<300,"R1",np.where(sec<600,"R2","R3")))
        ry.extend(rlab.tolist()); rp.extend(p["round4"].tolist())
    return {
        "duration":{m:binary_metrics(v["y"],v["p"]) for m,v in stores.items()},
        "round":multiclass_metrics(ry,np.asarray(rp),ROUND4),
    }


def train_duration_v03():
    hist=pd.read_csv(SEC_HIST)
    features=[c for c in hist.columns if c.endswith("_diff") and pd.api.types.is_numeric_dtype(hist[c])]
    metrics=duration_walk_forward(hist,features)

    cur=pd.read_csv(W7_SEC).merge(
        pd.read_csv(W7_RESULTS),
        on=["event_date","event_id","fighter_a","fighter_b"],
        how="inner",validate="one_to_one",suffixes=("","_result")
    )
    cur=orient_current(cur,features)
    current_train=cur[features+["duration_minutes"]].copy()
    hist_train=hist[features+["duration_minutes"]].copy()
    combined=pd.concat([hist_train,current_train],ignore_index=True)
    combined=duration_targets(combined)
    models=fit_hazards(combined,features)
    for end,m in zip(BOUNDARIES,models):
        if m is not None:
            joblib.dump(m,MODELS/f"dwcs_d_hazard_to_{end}s_v0_3.joblib")
    return {
        "version":"DWCS-D-v0.3-coherent-hazard",
        "round_version":"DWCS-R-v0.3-hazard-derived",
        "historical_walk_forward":metrics,
        "week7_rows_added":int(len(cur)),
        "training_rows":int(len(combined)),
        "boundaries_seconds":BOUNDARIES,
        "coherence_guarantee":"Survival probabilities are multiplied across ordered intervals, so O/U thresholds and round-start/distance markets are nested by construction.",
        "status":"SHADOW until walk-forward beats the prior/previous model on probability quality.",
    }


def main():
    winner=winner_retrain()
    method=train_method_v03()
    duration=train_duration_v03()
    summary={
        "event":"DWCS Season 10 Week 7 learning update",
        "graded_event_date":"2026-09-22",
        "winner":winner,
        "method":method,
        "duration_and_round":duration,
        "prospective_only_features":[
            "submission_access_quality_diff",
            "back_exposure_created_diff",
            "back_exposure_allowed_diff",
            "durability_after_clean_damage_diff",
            "failed_finish_cardio_cost_diff",
            "late_round_momentum_reversal_diff",
            "size_physicality_edge_diff",
            "decision_resilience_diff",
        ],
        "anti_overfit_rule":"Week 7 result-derived concepts are added to the prospective feature contract now, but are not retroactively fabricated for historical or Week 7 prefight rows. They begin receiving real values from Week 8 onward.",
    }
    (REPORTS/"week7_learning_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
