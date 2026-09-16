from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PAIR_FIELDS = [
    "age","height","reach","height_missing","reach_missing","southpaw","switch",
    "striker_style","grappler_style",
    "prior_dwcs_bouts","prior_dwcs_wins","prior_dwcs_win_pct",
    "prior_dwcs_finish_pct","prior_dwcs_ko_win_pct","prior_dwcs_sub_win_pct",
    "prior_dwcs_dec_win_pct","prior_dwcs_avg_duration","prior_dwcs_over25_rate",
    "prior_dwcs_stats_available","prior_dwcs_sig_landed_pm",
    "prior_dwcs_sig_attempted_pm","prior_dwcs_sig_accuracy",
    "prior_dwcs_td_landed_per15","prior_dwcs_td_attempted_per15",
    "prior_dwcs_td_accuracy","prior_dwcs_kd_per15","prior_dwcs_sub_attempts_per15",
    "global_career_bouts","global_career_wins","global_career_losses",
    "global_career_win_pct","global_recent3_win_pct","global_recent5_win_pct",
    "global_career_finish_pct","global_ko_win_pct","global_sub_win_pct",
    "global_decision_win_pct","global_finish_loss_pct","global_avg_opp_win_pct",
    "global_avg_opp_elo","global_quality_win_count","global_elo",
    "global_major_bouts","global_major_win_pct","global_win_streak",
    "global_days_since_last_fight",
]

def clean_name(x):
    return " ".join(str(x).casefold().strip().split())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--snapshots",type=Path,required=True)
    ap.add_argument("--matchups",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    snaps=pd.read_csv(args.snapshots)
    bouts=pd.read_csv(args.matchups)

    required={"fighter_name","as_of_date","history_established","provenance",*PAIR_FIELDS}
    missing=required-set(snaps.columns)
    if missing:
        raise ValueError(f"snapshot file missing columns: {sorted(missing)}")

    snaps["_key"]=snaps.fighter_name.map(clean_name)
    if snaps._key.duplicated().any():
        dup=snaps.loc[snaps._key.duplicated(keep=False),"fighter_name"].tolist()
        raise ValueError(f"duplicate fighter snapshots: {dup}")
    smap={str(r["_key"]):r for _,r in snaps.iterrows()}

    rows=[]
    for b in bouts.itertuples(index=False):
        ka,kb=clean_name(b.fighter_a),clean_name(b.fighter_b)
        if ka not in smap or kb not in smap:
            raise ValueError(f"missing snapshot for {b.fighter_a} vs {b.fighter_b}")
        a,bf=smap[ka],smap[kb]
        event_date=pd.Timestamp(b.event_date)
        for side,s in [("A",a),("B",bf)]:
            cutoff=pd.Timestamp(s.as_of_date)
            if cutoff>event_date:
                raise ValueError(f"{side} snapshot for {b.fighter_a} vs {b.fighter_b} is after event date")
        if int(a.history_established)!=1 or int(bf.history_established)!=1:
            gate=0
        else:
            gate=1

        row={
            "event_date":str(b.event_date),
            "event_id":str(b.event_id),
            "fighter_a":str(b.fighter_a),
            "fighter_b":str(b.fighter_b),
            "both_histories_established":gate,
            "fighter_a_snapshot_date":str(a.as_of_date),
            "fighter_b_snapshot_date":str(bf.as_of_date),
            "fighter_a_provenance":str(a.provenance),
            "fighter_b_provenance":str(bf.provenance),
        }
        for f in PAIR_FIELDS:
            av=pd.to_numeric(getattr(a,f),errors="coerce")
            bv=pd.to_numeric(getattr(bf,f),errors="coerce")
            row[f"{f}_diff"]=float(av-bv) if pd.notna(av) and pd.notna(bv) else np.nan
        rows.append(row)

    out=pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.out,index=False)
    print(f"Wrote {len(out)} current-card feature rows to {args.out}")

if __name__=="__main__":
    main()
