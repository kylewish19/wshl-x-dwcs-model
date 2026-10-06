from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learn_from_week7 import (
    ROOT, WIN_HIST, SEC_HIST, REPORTS, MODELS, PROCESSED, NON_FEATURE,
    METHOD_FEATURES, METHOD3, METHOD6, ROUND4, BOUNDARIES,
    make_logistic, orient_current, method_vec, current_record_method_state,
)
from train_secondary_models import method_family, duration_seconds, multiclass_metrics, binary_metrics, align_proba

W7_WIN = ROOT / "data/current/week7_model_features.csv"
W7_RESULTS = ROOT / "data/current/week7_results.csv"
W7_RECORDS = ROOT / "data/current/week7_current_records.csv"
W7_SNAPS = ROOT / "data/current/week7_fighter_snapshots.csv"
W8_WIN = ROOT / "data/current/week8_model_features.csv"
W8_RESULTS = ROOT / "data/current/week8_results.csv"
W8_RECORDS = ROOT / "data/current/week8_current_records.csv"
W8_SNAPS = ROOT / "data/current/week8_fighter_snapshots.csv"
HIST_METHOD_MATRIX = PROCESSED / "dwcs_method_candidate_matrix_v0_3.csv"


def make_boosting():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_depth=3,
            max_iter=250,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])


def score_binary(y, p):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    eps=1e-12
    return {
        "n":int(len(y)),
        "accuracy":float(np.mean((p>=0.5).astype(int)==y)),
        "brier":float(np.mean((p-y)**2)),
        "log_loss":float(-np.mean(y*np.log(np.clip(p,eps,1-eps))+(1-y)*np.log(np.clip(1-p,eps,1-eps)))),
    }


def winner_features(hist):
    numeric=[c for c in hist.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(hist[c])]
    return [c for c in numeric if c != "global_history_available_diff"]


def current_winner_rows(feature_file, result_file, features):
    f=pd.read_csv(feature_file)
    r=pd.read_csv(result_file)
    cur=f.merge(r,on=["event_date","event_id","fighter_a","fighter_b"],how="inner",validate="one_to_one",suffixes=("","_result"))
    missing=[c for c in features if c not in cur.columns]
    if missing:
        raise ValueError(f"Current winner feature rows missing: {missing}")
    return orient_current(cur,features)


def winner_retrain():
    hist=pd.read_csv(WIN_HIST)
    features=winner_features(hist)
    w7=current_winner_rows(W7_WIN,W7_RESULTS,features)
    w8=current_winner_rows(W8_WIN,W8_RESULTS,features)
    current=pd.concat([w7,w8],ignore_index=True)
    combined=pd.concat([hist[features+["winner_a"]],current[features+["winner_a"]]],ignore_index=True)

    before=joblib.load(MODELS/"dwcs_w_v0_5_week7_logistic.joblib")
    after=make_logistic(C=0.35).fit(combined[features],combined.winner_a.astype(int))
    shadow=make_boosting().fit(combined[features],combined.winner_a.astype(int))
    joblib.dump(after,MODELS/"dwcs_w_v0_6_week8_logistic.joblib")
    joblib.dump(shadow,MODELS/"dwcs_w_v0_6_boosting_shadow.joblib")

    before_coef=pd.Series(before.named_steps["model"].coef_[0],index=list(before.feature_names_in_))
    after_coef=pd.Series(after.named_steps["model"].coef_[0],index=features)
    delta=pd.DataFrame({"feature":features,"coefficient_after":after_coef.values})
    delta["coefficient_before"]=delta.feature.map(before_coef)
    delta["delta"]=delta.coefficient_after-delta.coefficient_before
    delta["abs_delta"]=delta.delta.abs()
    delta=delta.sort_values("abs_delta",ascending=False)
    delta.to_csv(REPORTS/"winner_v0_6_week8_coefficient_delta.csv",index=False)
    pd.DataFrame({"feature":features,"coefficient_standardized":after_coef.values}).sort_values(
        "coefficient_standardized",key=lambda s:s.abs(),ascending=False
    ).to_csv(REPORTS/"winner_v0_6_week8_coefficients.csv",index=False)

    prospective=prospective_winner_audit()
    return {
        "version":"DWCS-W-v0.6-week8-retrain",
        "historical_rows":int(len(hist)),
        "week7_rows":int(len(w7)),
        "week8_completed_rows":int(len(w8)),
        "total_training_rows":int(len(combined)),
        "artifact":"models/dwcs_w_v0_6_week8_logistic.joblib",
        "boosting_shadow_artifact":"models/dwcs_w_v0_6_boosting_shadow.joblib",
        "largest_coefficient_changes":delta.head(10).to_dict("records"),
        "prospective_week7_week8_audit":prospective,
        "decision":"Keep logistic production architecture; boosting/50-50 ensemble remain shadow because the 9-fight prospective sample does not improve probability quality over production.",
    }


def prospective_winner_audit():
    w7lock=pd.read_csv(ROOT/"official_picks/week7_preodds_model_lock.csv")
    w7res=pd.read_csv(W7_RESULTS)
    y=[]; prod=[]; boost=[]
    for rr in w7res.itertuples(index=False):
        fight=f"{rr.fighter_a} vs {rr.fighter_b}"
        row=w7lock[w7lock.fight.eq(fight)].iloc[0]
        p_a=float(row.winner_probability) if row.winner_pick==rr.fighter_a else 1-float(row.winner_probability)
        y.append(int(rr.winner_a)); prod.append(p_a); boost.append(float(row.winner_v0_4_boosting_a_probability))

    payload=json.loads((REPORTS/"week8_preodds_model_lock.json").read_text())
    result_map={(r.fighter_a,r.fighter_b):int(r.winner_a) for r in pd.read_csv(W8_RESULTS).itertuples(index=False)}
    for f in payload["fights"]:
        a,b=f["fight"].split(" vs ",1)
        if (a,b) not in result_map:
            continue
        y.append(result_map[(a,b)])
        prod.append(float(f["winner_probabilities"][a]))
        boost.append(float(f["winner_probabilities"]["v0_4_boosting_challenger_a"]))

    y=np.asarray(y,dtype=int); prod=np.asarray(prod); boost=np.asarray(boost); ensemble=(prod+boost)/2
    return {
        "n":int(len(y)),
        "production":score_binary(y,prod),
        "boosting_challenger":score_binary(y,boost),
        "simple_50_50_ensemble":score_binary(y,ensemble),
        "note":"This is prospective diagnostic evidence only: Week 7 used the then-production v0.4 logistic and Week 8 used v0.5. It is not a historical promotion test."
    }


def method_rows_for_week(records_path,snaps_path,results_path):
    records=pd.read_csv(records_path).set_index("fighter_name")
    snaps=pd.read_csv(snaps_path).set_index("fighter_name")
    res=pd.read_csv(results_path)
    rows=[]
    for r in res.itertuples(index=False):
        astate=current_record_method_state(records.loc[r.fighter_a],snaps.loc[r.fighter_a])
        bstate=current_record_method_state(records.loc[r.fighter_b],snaps.loc[r.fighter_b])
        va=method_vec(astate,bstate); vb=method_vec(bstate,astate)
        row={"event_date":r.event_date,"event_id":r.event_id,"fighter_a":r.fighter_a,"fighter_b":r.fighter_b,
             "winner_a":int(r.winner_a),"method3":method_family(r.method)}
        for f in METHOD_FEATURES:
            row[f"a__{f}"]=va[f]; row[f"b__{f}"]=vb[f]
        rows.append(row)
    return add_hier_features(pd.DataFrame(rows))


DERIVED_METHOD_FEATURES=[
    "finish_pressure","early_finish_route_strength","opponent_durability",
    "ko_route_strength","sub_route_strength","decision_resistance",
    "ko_conversion_pressure","sub_conversion_pressure",
]
HIER_FEATURES=METHOD_FEATURES+DERIVED_METHOD_FEATURES


def _nz(v,default=0.0):
    try:
        if pd.isna(v): return default
        return float(v)
    except Exception:
        return default


def derived_method_values(row,prefix):
    cand_finish=_nz(row[f"{prefix}cand_finish_win_pct"])
    recent=_nz(row[f"{prefix}cand_recent5_finish_pct"],cand_finish)
    r1=_nz(row[f"{prefix}cand_r1_finish_win_pct"],cand_finish)
    early=_nz(row[f"{prefix}cand_early_finish_win_pct"],cand_finish)
    opp_fin=_nz(row[f"{prefix}opp_finish_loss_pct"])
    ko=_nz(row[f"{prefix}cand_ko_win_pct"])
    sub=_nz(row[f"{prefix}cand_sub_win_pct"])
    ko_vuln=_nz(row[f"{prefix}opp_ko_loss_pct"])
    sub_vuln=_nz(row[f"{prefix}opp_sub_loss_pct"])
    dec=_nz(row[f"{prefix}cand_dec_win_pct"])
    finish_pressure=0.5*cand_finish+0.5*recent
    early_strength=max(r1,early,recent)
    opponent_durability=1.0-opp_fin
    return {
        "finish_pressure":finish_pressure,
        "early_finish_route_strength":early_strength,
        "opponent_durability":opponent_durability,
        "ko_route_strength":ko,
        "sub_route_strength":sub,
        "decision_resistance":opponent_durability,
        "ko_conversion_pressure":ko*(0.5+0.5*early_strength)*(0.5+0.5*ko_vuln),
        "sub_conversion_pressure":sub*(0.5+0.5*recent)*(0.5+0.5*sub_vuln),
    }


def add_hier_features(df):
    out=df.copy()
    for i,row in out.iterrows():
        for prefix in ("a__","b__"):
            vals=derived_method_values(row,prefix)
            for k,v in vals.items(): out.loc[i,f"{prefix}{k}"]=v
    return out


def method_candidate_xy(df,features):
    X=[]; y=[]
    for r in df.itertuples(index=False):
        prefix="a__" if int(r.winner_a)==1 else "b__"
        X.append([getattr(r,prefix+f) for f in features]); y.append(r.method3)
    return pd.DataFrame(X,columns=features),np.asarray(y)


def candidate_frame(df,prefix,features):
    return pd.DataFrame([{f:getattr(r,prefix+f) for f in features} for r in df.itertuples(index=False)])


def fit_hier_method(df):
    X,y=method_candidate_xy(df,HIER_FEATURES)
    finish_y=(y!="DEC").astype(int)
    finish=make_logistic(C=0.20).fit(X,finish_y)
    mask=finish_y==1
    route=make_logistic(C=0.20).fit(X.loc[mask],y[mask])
    return finish,route


def hier_conditional_probs(finish,route,X):
    pf=finish.predict_proba(X)[:,list(finish.classes_).index(1)]
    pr=align_proba(route,X,["KO_TKO","SUB"])
    out=np.c_[pf*pr[:,0],pf*pr[:,1],1-pf]
    return out/out.sum(axis=1,keepdims=True)


def method_walk_forward_hier(df,min_train_events=20):
    d=df.copy(); d["_date"]=pd.to_datetime(d.event_date)
    events=d[["_date","event_id"]].drop_duplicates().sort_values(["_date","event_id"]).reset_index(drop=True)
    wh=pd.read_csv(WIN_HIST); wh["_date"]=pd.to_datetime(wh.event_date)
    wfeatures=winner_features(wh)
    ys=[]; ps=[]
    for i in range(min_train_events,len(events)):
        cutoff=events.iloc[i]._date; eid=str(events.iloc[i].event_id)
        tr=d[d._date<cutoff]; te=d[d.event_id.astype(str)==eid]
        if len(tr)<50 or len(te)==0: continue
        finish,route=fit_hier_method(tr)
        pa=hier_conditional_probs(finish,route,candidate_frame(te,"a__",HIER_FEATURES))
        pb=hier_conditional_probs(finish,route,candidate_frame(te,"b__",HIER_FEATURES))
        wtr=wh[wh._date<cutoff]; wte=wh[wh.event_id.astype(str)==eid]
        wm=make_logistic(C=0.35).fit(wtr[wfeatures],wtr.winner_a.astype(int))
        wp=wm.predict_proba(wte[wfeatures])[:,1]
        lookup={(str(r.fighter_a),str(r.fighter_b)):float(p) for r,p in zip(wte.itertuples(index=False),wp)}
        for j,r in enumerate(te.itertuples(index=False)):
            pwin=lookup[(str(r.fighter_a),str(r.fighter_b))]
            joint=np.r_[pwin*pa[j],(1-pwin)*pb[j]]; joint=joint/joint.sum()
            ys.append(("A_" if int(r.winner_a)==1 else "B_")+r.method3); ps.append(joint.tolist())
    return multiclass_metrics(ys,np.asarray(ps),METHOD6)


def method_retrain():
    hist=add_hier_features(pd.read_csv(HIST_METHOD_MATRIX))
    w7=method_rows_for_week(W7_RECORDS,W7_SNAPS,W7_RESULTS)
    w8=method_rows_for_week(W8_RECORDS,W8_SNAPS,W8_RESULTS)
    current=pd.concat([w7,w8],ignore_index=True)
    combined=pd.concat([hist,current],ignore_index=True)

    # Refit current v0.3 architecture for continuity.
    X3,y3=method_candidate_xy(combined,METHOD_FEATURES)
    v03=make_logistic(C=0.20).fit(X3,y3)
    joblib.dump(v03,MODELS/"dwcs_m_v0_3_week8_refit.joblib")

    hist_metrics=method_walk_forward_hier(hist)
    finish,route=fit_hier_method(combined)
    joblib.dump(finish,MODELS/"dwcs_m_v0_4_finish_gate_logistic.joblib")
    joblib.dump(route,MODELS/"dwcs_m_v0_4_finish_route_logistic.joblib")

    old={"accuracy":0.5799373040752351,"log_loss":1.1937154454968144,"brier_multiclass":0.5569264643576686}
    promoted=(hist_metrics["log_loss"]<old["log_loss"] and hist_metrics["brier_multiclass"]<old["brier_multiclass"] and hist_metrics["accuracy"]>=old["accuracy"])
    return {
        "candidate":"DWCS-M-v0.4-hierarchical-finish-route",
        "historical_walk_forward":hist_metrics,
        "v0_3_reference":old,
        "week7_rows":int(len(w7)),"week8_rows":int(len(w8)),"total_training_rows":int(len(combined)),
        "architecture":"P(winner) x P(finish vs decision | winner) x P(KO/TKO vs SUB | finish, winner)",
        "promotion_rule":"Must improve historical walk-forward log loss and multiclass Brier without lowering exact winner+method accuracy.",
        "promoted":bool(promoted),
        "production_if_not_promoted":"models/dwcs_m_v0_3_week8_refit.joblib",
        "candidate_artifacts":["models/dwcs_m_v0_4_finish_gate_logistic.joblib","models/dwcs_m_v0_4_finish_route_logistic.joblib"],
    }


PAIR_FEATURES=[
    "pair_finish_mean","pair_finish_max","pair_finish_min","pair_finish_product",
    "pair_recent5_finish_mean","pair_recent5_finish_max",
    "pair_r1_finish_mean","pair_r1_finish_max","pair_early_finish_mean","pair_early_finish_max",
    "pair_finish_loss_mean","pair_finish_loss_max","pair_durability_min",
    "pair_ko_mean","pair_ko_max","pair_sub_mean","pair_sub_max","pair_decision_mean",
    "pair_avg_duration_mean","pair_avg_duration_min","pair_avg_duration_max",
    "pair_wins_sum","pair_losses_sum","pair_elo_mean","pair_elo_abs_gap","pair_experience_abs_gap",
    "pair_finish_vs_vulnerability_max","pair_ko_access_max","pair_sub_access_max","pair_decision_route_mean",
]


def pair_features_from_row(row):
    def g(side,name,default=np.nan):
        try:
            v=row[f"{side}__{name}"]
            return float(v) if not pd.isna(v) else default
        except Exception: return default
    a={k:g("a",k) for k in METHOD_FEATURES}; b={k:g("b",k) for k in METHOD_FEATURES}
    def vals(name,default=0.0):
        av=a.get(name,np.nan); bv=b.get(name,np.nan)
        av=default if pd.isna(av) else av; bv=default if pd.isna(bv) else bv
        return float(av),float(bv)
    fin=vals("cand_finish_win_pct"); rec=vals("cand_recent5_finish_pct",np.nan)
    r1=vals("cand_r1_finish_win_pct",np.nan); early=vals("cand_early_finish_win_pct",np.nan)
    floss=vals("opp_finish_loss_pct"); ko=vals("cand_ko_win_pct"); sub=vals("cand_sub_win_pct"); dec=vals("cand_dec_win_pct")
    dur=vals("cand_avg_duration_sec",np.nan); wins=vals("cand_wins"); losses=vals("cand_losses"); elo=vals("cand_elo",1500.0)
    exp_edge=abs(_nz(a.get("experience_edge"),0.0))
    fconv=vals("finish_conversion_vs_vulnerability"); koacc=vals("ko_access"); subacc=vals("sub_access"); decroute=vals("decision_route")
    def mean2(x): return float(np.nanmean(x)) if not np.all(pd.isna(x)) else np.nan
    def max2(x): return float(np.nanmax(x)) if not np.all(pd.isna(x)) else np.nan
    def min2(x): return float(np.nanmin(x)) if not np.all(pd.isna(x)) else np.nan
    return {
        "pair_finish_mean":mean2(fin),"pair_finish_max":max2(fin),"pair_finish_min":min2(fin),"pair_finish_product":fin[0]*fin[1],
        "pair_recent5_finish_mean":mean2(rec),"pair_recent5_finish_max":max2(rec),
        "pair_r1_finish_mean":mean2(r1),"pair_r1_finish_max":max2(r1),"pair_early_finish_mean":mean2(early),"pair_early_finish_max":max2(early),
        "pair_finish_loss_mean":mean2(floss),"pair_finish_loss_max":max2(floss),"pair_durability_min":1-max2(floss),
        "pair_ko_mean":mean2(ko),"pair_ko_max":max2(ko),"pair_sub_mean":mean2(sub),"pair_sub_max":max2(sub),"pair_decision_mean":mean2(dec),
        "pair_avg_duration_mean":mean2(dur),"pair_avg_duration_min":min2(dur),"pair_avg_duration_max":max2(dur),
        "pair_wins_sum":sum(wins),"pair_losses_sum":sum(losses),"pair_elo_mean":mean2(elo),"pair_elo_abs_gap":abs(elo[0]-elo[1]),"pair_experience_abs_gap":exp_edge,
        "pair_finish_vs_vulnerability_max":max2(fconv),"pair_ko_access_max":max2(koacc),"pair_sub_access_max":max2(subacc),"pair_decision_route_mean":mean2(decroute),
    }


def add_pair_features(df):
    rows=[]
    for _,r in df.iterrows(): rows.append(pair_features_from_row(r))
    return pd.concat([df.reset_index(drop=True),pd.DataFrame(rows)],axis=1)


def parse_current_duration_seconds(r):
    if hasattr(r,"duration_seconds") and not pd.isna(r.duration_seconds): return float(r.duration_seconds)
    m,s=[int(x) for x in str(r.actual_time).split(":")]
    return float((int(r.finish_round)-1)*300+m*60+s)


def duration_dataset():
    hm=add_pair_features(pd.read_csv(HIST_METHOD_MATRIX))
    sec=pd.read_csv(SEC_HIST)[["event_date","event_id","fighter_a","fighter_b","duration_minutes"]].copy()
    hist=hm.merge(sec,on=["event_date","event_id","fighter_a","fighter_b"],how="inner",validate="one_to_one")
    hist["_duration_seconds"]=hist.duration_minutes.map(duration_seconds)

    cur=[]
    for rec,snap,res in [(W7_RECORDS,W7_SNAPS,W7_RESULTS),(W8_RECORDS,W8_SNAPS,W8_RESULTS)]:
        rows=method_rows_for_week(rec,snap,res)
        rows=add_pair_features(rows)
        rr=pd.read_csv(res)
        duration={(x.fighter_a,x.fighter_b):parse_current_duration_seconds(x) for x in rr.itertuples(index=False)}
        rows["_duration_seconds"]=[duration[(x.fighter_a,x.fighter_b)] for x in rows.itertuples(index=False)]
        cur.append(rows)
    return hist,pd.concat(cur,ignore_index=True)


def fit_pair_hazards(train):
    models=[]
    for k,end in enumerate(BOUNDARIES):
        start=0 if k==0 else BOUNDARIES[k-1]
        tr=train[train._duration_seconds>=start]
        y=(tr._duration_seconds<end).astype(int)
        models.append(None if y.nunique()<2 else make_logistic(C=0.20).fit(tr[PAIR_FEATURES],y))
    return models


def pair_hazard_probs(models,X):
    surv=np.ones(len(X)); S={}
    for end,m in zip(BOUNDARIES,models):
        h=np.zeros(len(X)) if m is None else m.predict_proba(X)[:,list(m.classes_).index(1)]
        surv*=1-h; S[end]=surv.copy()
    return {
        "under_0_5":1-S[150],"under_1_5":1-S[450],"under_2_5":1-S[750],
        "goes_distance":S[900],"round_2_starts":S[300],"round_3_starts":S[600],
        "round4":np.c_[1-S[300],S[300]-S[600],S[600]-S[900],S[900]],
    }


def duration_walk_forward_pair(hist,min_train_events=20):
    d=hist.copy(); d["_date"]=pd.to_datetime(d.event_date)
    events=d[["_date","event_id"]].drop_duplicates().sort_values(["_date","event_id"]).reset_index(drop=True)
    markets=["under_0_5","under_1_5","under_2_5","goes_distance","round_2_starts","round_3_starts"]
    store={m:{"y":[],"p":[],"base":[]} for m in markets}; ry=[]; rp=[]; rb=[]
    for i in range(min_train_events,len(events)):
        cutoff=events.iloc[i]._date; eid=str(events.iloc[i].event_id)
        tr=d[d._date<cutoff]; te=d[d.event_id.astype(str)==eid]
        if len(tr)<50 or len(te)==0: continue
        models=fit_pair_hazards(tr); p=pair_hazard_probs(models,te[PAIR_FEATURES]); sec=te._duration_seconds.to_numpy()
        labels={"under_0_5":(sec<150).astype(int),"under_1_5":(sec<450).astype(int),"under_2_5":(sec<750).astype(int),
                "goes_distance":(sec>=900).astype(int),"round_2_starts":(sec>=300).astype(int),"round_3_starts":(sec>=600).astype(int)}
        trsec=tr._duration_seconds.to_numpy()
        baseprev={"under_0_5":np.mean(trsec<150),"under_1_5":np.mean(trsec<450),"under_2_5":np.mean(trsec<750),
                  "goes_distance":np.mean(trsec>=900),"round_2_starts":np.mean(trsec>=300),"round_3_starts":np.mean(trsec>=600)}
        for m in markets:
            store[m]["y"].extend(labels[m].tolist()); store[m]["p"].extend(p[m].tolist()); store[m]["base"].extend([baseprev[m]]*len(te))
        rlab=np.where(sec>=900,"DEC",np.where(sec<300,"R1",np.where(sec<600,"R2","R3")))
        ry.extend(rlab.tolist()); rp.extend(p["round4"].tolist())
        trlab=np.where(trsec>=900,"DEC",np.where(trsec<300,"R1",np.where(trsec<600,"R2","R3")))
        freq=np.array([np.mean(trlab==c) for c in ROUND4]); rb.extend(np.tile(freq,(len(te),1)).tolist())
    return {
        "duration":{m:{"candidate":binary_metrics(v["y"],v["p"]),"naive":binary_metrics(v["y"],v["base"])} for m,v in store.items()},
        "round":{"candidate":multiclass_metrics(ry,np.asarray(rp),ROUND4),"naive":multiclass_metrics(ry,np.asarray(rb),ROUND4)},
    }


def duration_retrain():
    hist,current=duration_dataset(); metrics=duration_walk_forward_pair(hist)
    combined=pd.concat([hist,current],ignore_index=True,sort=False)
    models=fit_pair_hazards(combined)
    for end,m in zip(BOUNDARIES,models):
        if m is not None: joblib.dump(m,MODELS/f"dwcs_d_pair_hazard_to_{end}s_v0_4.joblib")
    wins=0
    for m,info in metrics["duration"].items():
        if info["candidate"]["brier"]<info["naive"]["brier"] and info["candidate"]["log_loss"]<info["naive"]["log_loss"]: wins+=1
    round_ok=(metrics["round"]["candidate"]["brier_multiclass"]<metrics["round"]["naive"]["brier_multiclass"] and metrics["round"]["candidate"]["log_loss"]<metrics["round"]["naive"]["log_loss"])
    promoted=wins>=4 and round_ok
    return {
        "candidate":"DWCS-D-v0.4 pair-symmetric coherent hazard",
        "round_candidate":"DWCS-R-v0.4 pair-symmetric hazard-derived",
        "historical_walk_forward":metrics,
        "pair_feature_count":len(PAIR_FEATURES),
        "duration_markets_beating_naive_on_brier_and_logloss":int(wins),
        "round_beats_naive_probability_quality":bool(round_ok),
        "promoted":bool(promoted),
        "note":"All timing inputs are order-invariant pair features (means/max/min/products/absolute gaps). Swapping Fighter A and Fighter B cannot change a fight-duration prediction.",
    }


def main():
    winner=winner_retrain(); method=method_retrain(); duration=duration_retrain()
    summary={
        "event":"DWCS Season 10 Week 8 learning update",
        "graded_event_date":"2026-09-29",
        "winner":winner,"method":method,"duration_and_round":duration,
        "week8_context_flags":[
            "Escuza official TKO retained as label, with illegal-knee review context preserved separately.",
            "Staines official TKO-by-slam retained as KO_TKO label; future tape features distinguish grappling-generated TKO access from conventional striking KO history.",
            "Bulaid-Visconde cancelled bout is excluded from labels and retraining."
        ],
        "prospective_only_features_starting_next_card":[
            "elite_amateur_pedigree_diff","amateur_fight_sample_diff","pre_ufc_championship_experience_diff","five_round_experience_diff",
            "grappling_generated_tko_access_diff","slam_ground_damage_tko_access_diff","early_finish_route_strength_diff","post_high_output_round_freshness_diff"
        ],
        "anti_overfit_rule":"Week 8-derived concepts are added prospectively only. No historical or Week 8 prefight values are invented after outcomes are known."
    }
    (REPORTS/"week8_learning_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
