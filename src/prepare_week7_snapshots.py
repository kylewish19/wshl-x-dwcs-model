from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from augment_regional_career import build_timelines, prefight_state, expected

# Week 7 form integrity fix: current pro form overrides mixed-source recent history.\nROOT=Path(__file__).resolve().parents[1]
CURRENT=ROOT/"data"/"current"
RAW=ROOT/"data"/"raw"
REPORTS=ROOT/"reports"

MATCHUPS=CURRENT/"week7_matchups.csv"
RECORDS=CURRENT/"week7_current_records.csv"
POSTCUT=CURRENT/"week7_postcutoff_fights.csv"
OUT=CURRENT/"week7_fighter_snapshots.csv"
OUT_REPORT=REPORTS/"week7_snapshot_coverage.json"

def norm_name(v):
    s=unicodedata.normalize("NFKD",str(v or ""))
    s="".join(c for c in s if not unicodedata.combining(c))
    s=s.casefold().replace("’","'")
    s=re.sub(r"[^a-z0-9]+"," ",s)
    return " ".join(s.split())

def ratio(n,d):
    return float(n)/float(d) if d else 0.0

def safe(v,default=np.nan):
    try:
        if pd.isna(v) or str(v).strip()=="":
            return default
        return float(v)
    except Exception:
        return default

def age_on(dob,event_date):
    if dob is None or pd.isna(dob) or str(dob).strip()=="":
        return np.nan
    return (pd.Timestamp(event_date)-pd.Timestamp(dob)).days/365.2425

def style_flags(style):
    s=str(style or "").casefold()
    return (
        int(any(k in s for k in ["strik","boxing","kickbox","muay"])),
        int(any(k in s for k in ["grap","wrest","jiu","judo","sambo"]))
    )

def stance_flags(stance):
    s=str(stance or "").casefold()
    return int("southpaw" in s),int("switch" in s)

def dwcs_empty():
    return {
        "prior_dwcs_bouts":0.0,"prior_dwcs_wins":0.0,
        "prior_dwcs_win_pct":0.0,"prior_dwcs_finish_pct":0.0,
        "prior_dwcs_ko_win_pct":0.0,"prior_dwcs_sub_win_pct":0.0,
        "prior_dwcs_dec_win_pct":0.0,"prior_dwcs_avg_duration":0.0,
        "prior_dwcs_over25_rate":0.0,"prior_dwcs_stats_available":0.0,
        "prior_dwcs_sig_landed_pm":0.0,"prior_dwcs_sig_attempted_pm":0.0,
        "prior_dwcs_sig_accuracy":0.0,"prior_dwcs_td_landed_per15":0.0,
        "prior_dwcs_td_attempted_per15":0.0,"prior_dwcs_td_accuracy":0.0,
        "prior_dwcs_kd_per15":0.0,"prior_dwcs_sub_attempts_per15":0.0,
    }

def build_prior_dwcs():
    hist=pd.read_csv(RAW/"espn_dwcs_fighter_history_2017_2025.csv")
    stats=pd.read_csv(RAW/"espn_dwcs_fighter_stats_2017_2025.csv")
    stats_map={str(r["uid"]):r for _,r in stats.iterrows()}
    by=defaultdict(list)
    for _,r in hist.iterrows():
        by[norm_name(r["fighter_name"])].append(r)

    out={}
    for key,rows in by.items():
        bouts=wins=finishes=ko=sub=dec=over25=0
        dur_sum=0.0
        stat_bouts=0
        stat_minutes=sig_l=sig_a=td_l=td_a=kd=subatt=0.0
        for r in sorted(rows,key=lambda x:str(x["event_date"])):
            bouts+=1
            won=str(r["fight_result"]).upper()=="W"
            method=str(r["fight_result_type"]).upper()
            dur=safe(r["fight_duration"],0.0)
            dur_sum+=dur
            over25+=int(bool(r["over_2_5"]))
            if won:
                wins+=1
                if method in {"KO-TKO","SUBMISSION","SUB"}:
                    finishes+=1
                if method=="KO-TKO": ko+=1
                elif method in {"SUBMISSION","SUB"}: sub+=1
                elif method.startswith("DEC"): dec+=1
            sr=stats_map.get(str(r["uid"]))
            if sr is not None and dur>0:
                stat_bouts+=1
                stat_minutes+=dur
                for col,var in [("SSL","sig_l"),("SSA","sig_a"),("TDL","td_l"),
                                ("TDA","td_a"),("KD","kd"),("SM","subatt")]:
                    val=safe(sr.get(col,0),0.0)
                    if col=="SSL": sig_l+=val
                    elif col=="SSA": sig_a+=val
                    elif col=="TDL": td_l+=val
                    elif col=="TDA": td_a+=val
                    elif col=="KD": kd+=val
                    elif col=="SM": subatt+=val
        out[key]={
            "prior_dwcs_bouts":float(bouts),
            "prior_dwcs_wins":float(wins),
            "prior_dwcs_win_pct":ratio(wins,bouts),
            "prior_dwcs_finish_pct":ratio(finishes,bouts),
            "prior_dwcs_ko_win_pct":ratio(ko,bouts),
            "prior_dwcs_sub_win_pct":ratio(sub,bouts),
            "prior_dwcs_dec_win_pct":ratio(dec,bouts),
            "prior_dwcs_avg_duration":ratio(dur_sum,bouts),
            "prior_dwcs_over25_rate":ratio(over25,bouts),
            "prior_dwcs_stats_available":float(stat_bouts>0),
            "prior_dwcs_sig_landed_pm":ratio(sig_l,stat_minutes),
            "prior_dwcs_sig_attempted_pm":ratio(sig_a,stat_minutes),
            "prior_dwcs_sig_accuracy":ratio(sig_l,sig_a),
            "prior_dwcs_td_landed_per15":15*ratio(td_l,stat_minutes),
            "prior_dwcs_td_attempted_per15":15*ratio(td_a,stat_minutes),
            "prior_dwcs_td_accuracy":ratio(td_l,td_a),
            "prior_dwcs_kd_per15":15*ratio(kd,stat_minutes),
            "prior_dwcs_sub_attempts_per15":15*ratio(subatt,stat_minutes),
        }
    return out

def current_recent_results(con,targets,postcut):
    f=con.execute("""
        SELECT event_date,fighter_1,fighter_2,winner
        FROM fights_career_longitudinal
        ORDER BY event_date
    """).fetchdf()
    results=defaultdict(list)
    for r in f.itertuples(index=False):
        a,b,w=norm_name(r.fighter_1),norm_name(r.fighter_2),norm_name(r.winner)
        date=pd.Timestamp(r.event_date)
        if a in targets and w in {a,b}:
            results[a].append((date,1 if w==a else 0))
        if b in targets and w in {a,b}:
            results[b].append((date,1 if w==b else 0))
    for r in postcut.itertuples(index=False):
        k=norm_name(r.fighter_name)
        if k in targets:
            results[k].append((pd.Timestamp(r.event_date),1 if str(r.result).upper()=="W" else 0))
    for k in results:
        results[k].sort(key=lambda x:x[0])
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--database",type=Path,required=True)
    args=ap.parse_args()

    matchups=pd.read_csv(MATCHUPS)
    records=pd.read_csv(RECORDS,keep_default_na=False)
    postcut=pd.read_csv(POSTCUT,keep_default_na=False)
    event_date=pd.Timestamp(matchups.event_date.iloc[0])

    target_names=sorted(set(matchups.fighter_a)|set(matchups.fighter_b))
    targets={norm_name(x) for x in target_names}

    timelines,source_fights,source_fighters=build_timelines(args.database,(event_date+pd.Timedelta(days=1)).strftime("%Y-%m-%d"))

    con=duckdb.connect(str(args.database),read_only=True)
    master=con.execute("SELECT fighter_name,dob,height_cm,reach_cm,stance FROM fighters_master").fetchdf()
    master["_key"]=master.fighter_name.map(norm_name)
    # Prefer rows with more populated physical fields.
    master["_coverage"]=master[["dob","height_cm","reach_cm","stance"]].notna().sum(axis=1)
    master=master.sort_values("_coverage",ascending=False).drop_duplicates("_key")
    master_map={str(r["_key"]):r for _,r in master.iterrows()}
    recents=current_recent_results(con,targets,postcut)
    con.close()

    attrs=pd.read_csv(RAW/"espn_dwcs_fighter_attributes_2017_2025.csv",keep_default_na=False)
    attrs["_key"]=attrs.name.map(norm_name)
    attrs=attrs.drop_duplicates("_key")
    attr_map={str(r["_key"]):r for _,r in attrs.iterrows()}
    dwcs=build_prior_dwcs()

    record_map={norm_name(r.fighter_name):r for r in records.itertuples(index=False)}
    post_map=defaultdict(list)
    for r in postcut.itertuples(index=False):
        post_map[norm_name(r.fighter_name)].append(r)
    for k in post_map:
        post_map[k].sort(key=lambda r:r.event_date)

    rows=[]
    coverage=[]
    for name in target_names:
        key=norm_name(name)
        rec=record_map[key]
        base=prefight_state(timelines,name,event_date)
        hist_ok=base is not None and str(rec.record_status).upper()=="RECONCILED"

        # Current aggregate career state comes from the reconciled live profile.
        wins=int(rec.wins); losses=int(rec.losses); nc=int(rec.nc)
        decided=wins+losses
        career_bouts=wins+losses+nc

        if base is None:
            avg_opp_wp=0.5; avg_opp_elo=1500.0; quality=0.0; elo=1500.0
            major_bouts=0.0; major_wins=0.0
            base_bouts=0
        else:
            base_bouts=int(base["career_bouts"])
            avg_opp_wp=float(base["avg_opp_win_pct"])
            avg_opp_elo=float(base["avg_opp_elo"])
            quality=float(base["quality_win_count"])
            elo=float(base["elo"])
            major_bouts=float(base["major_bouts"])
            major_wins=float(base["major_win_pct"])*major_bouts

        opp_wp_sum=avg_opp_wp*base_bouts
        opp_elo_sum=avg_opp_elo*base_bouts
        opp_n=base_bouts

        for u in post_map.get(key,[]):
            opp=prefight_state(timelines,u.opponent_name,pd.Timestamp(u.event_date))
            if opp is None:
                owp=0.5; oelo=1500.0; obouts=0
            else:
                owp=float(opp["career_win_pct"])
                oelo=float(opp["elo"])
                obouts=int(opp["career_bouts"])
            opp_wp_sum+=owp; opp_elo_sum+=oelo; opp_n+=1
            if str(u.result).upper()=="W" and obouts>=5 and (owp>=0.65 or oelo>=1550):
                quality+=1
            if int(u.is_major_org)==1:
                major_bouts+=1
                if str(u.result).upper()=="W": major_wins+=1
            exp=expected(elo,oelo)
            actual=1.0 if str(u.result).upper()=="W" else 0.0
            elo+=32.0*(actual-exp)

        # Current-form inputs are reconciled from the fighter's professional
        # record. The global database can contain amateur/duplicate history,
        # which is useful for broader context but must not define pro win streak.
        recent3=safe(getattr(rec,"recent3_win_pct",""),np.nan)
        recent5=safe(getattr(rec,"recent5_win_pct",""),np.nan)
        streak=safe(getattr(rec,"current_win_streak",""),np.nan)
        if not (np.isfinite(recent3) and np.isfinite(recent5) and np.isfinite(streak)):
            seq=[v for _,v in recents.get(key,[])]
            recent3=ratio(sum(seq[-3:]),len(seq[-3:]))
            recent5=ratio(sum(seq[-5:]),len(seq[-5:]))
            streak=0
            for v in reversed(seq):
                if v==1: streak+=1
                else: break
        # Integrity guard: a professional win streak cannot exceed total pro wins.
        streak=min(float(streak),float(wins))

        # Static physical/style state.
        gm=master_map.get(key)
        dob=str(rec.dob).strip() or ("" if gm is None or pd.isna(gm.dob) else str(gm.dob))
        height=safe(rec.height_cm,np.nan)
        if not np.isfinite(height) and gm is not None: height=safe(gm.height_cm,np.nan)
        reach=safe(gm.reach_cm,np.nan) if gm is not None else np.nan
        stance="" if gm is None or pd.isna(gm.stance) else str(gm.stance)
        southpaw,switch=stance_flags(stance)
        ea=attr_map.get(key)
        style="" if ea is None else str(ea.style)
        striker,grappler=style_flags(style)

        last_date=pd.Timestamp(rec.last_fight_date)
        current_global={
            "global_career_bouts":float(career_bouts),
            "global_career_wins":float(wins),
            "global_career_losses":float(losses),
            "global_career_win_pct":ratio(wins,decided),
            "global_recent3_win_pct":recent3,
            "global_recent5_win_pct":recent5,
            "global_career_finish_pct":ratio(int(rec.ko_wins)+int(rec.sub_wins),wins),
            "global_ko_win_pct":ratio(int(rec.ko_wins),wins),
            "global_sub_win_pct":ratio(int(rec.sub_wins),wins),
            "global_decision_win_pct":ratio(int(rec.dec_wins),wins),
            "global_finish_loss_pct":ratio(int(rec.ko_losses)+int(rec.sub_losses),losses),
            "global_avg_opp_win_pct":ratio(opp_wp_sum,opp_n) if opp_n else 0.5,
            "global_avg_opp_elo":ratio(opp_elo_sum,opp_n) if opp_n else 1500.0,
            "global_quality_win_count":quality,
            "global_elo":elo,
            "global_major_bouts":major_bouts,
            "global_major_win_pct":ratio(major_wins,major_bouts),
            "global_win_streak":float(streak),
            "global_days_since_last_fight":float((event_date-last_date).days),
        }

        row={
            "fighter_name":name,
            "as_of_date":str(rec.as_of_date),
            "history_established":int(hist_ok),
            "provenance":"MMA Global career history + Sherdog current record/post-cutoff + ESPN prior DWCS",
            "age":age_on(dob,event_date),
            "height":height,
            "reach":reach,
            "height_missing":float(not np.isfinite(height)),
            "reach_missing":float(not np.isfinite(reach)),
            "southpaw":float(southpaw),
            "switch":float(switch),
            "striker_style":float(striker),
            "grappler_style":float(grappler),
        }
        row.update(dwcs.get(key,dwcs_empty()))
        row.update(current_global)
        rows.append(row)
        coverage.append({
            "fighter":name,
            "history_established":int(hist_ok),
            "global_history_matched":int(base is not None),
            "current_record_reconciled":int(str(rec.record_status).upper()=="RECONCILED"),
            "postcut_updates":len(post_map.get(key,[])),
            "prior_dwcs_bouts":row["prior_dwcs_bouts"],
        })

    out=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT,index=False)

    cov=pd.DataFrame(coverage)
    bout_gate=[]
    for b in matchups.itertuples(index=False):
        ak=int(cov.loc[cov.fighter.eq(b.fighter_a),"history_established"].iloc[0])
        bk=int(cov.loc[cov.fighter.eq(b.fighter_b),"history_established"].iloc[0])
        bout_gate.append({"fighter_a":b.fighter_a,"fighter_b":b.fighter_b,"both_histories_established":int(ak and bk)})
    report={
        "event":"DWCS Season 10 Week 7",
        "event_date":str(event_date.date()),
        "source_global_fights_loaded":source_fights,
        "source_global_unique_fighters":source_fighters,
        "fighter_coverage":coverage,
        "bout_coverage":bout_gate,
        "primary_v0_4_bouts":sum(x["both_histories_established"] for x in bout_gate),
        "total_bouts":len(bout_gate),
    }
    REPORTS.mkdir(parents=True,exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report,indent=2))
    print(out.to_string(index=False))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
