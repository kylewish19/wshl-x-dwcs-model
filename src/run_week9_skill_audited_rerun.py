from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/"reports"; DIAG=ROOT/"diagnostics"; MODELS=ROOT/"models"; OFFICIAL=ROOT/"official_picks"
LOCK_JSON=REPORTS/"week9_preodds_model_lock.json"
LOCK_CSV=OFFICIAL/"week9_preodds_model_lock.csv"
MOD=DIAG/"week9_v0_7_modular_rerun.csv"
TAPE=ROOT/"data/current/week9_tape_features.csv"

MODULES=["resume_form","striking","grappling","durability","cardio_decision","physical_quality"]
JOINT=["A_KO_TKO","A_SUB","A_DEC","B_KO_TKO","B_SUB","B_DEC"]
ROUNDS=["R1","R2","R3","DEC"]


def audit_policy(raw_prob,coverage_status,direct_coverage,support,spread):
    # This is a transparent prospective betting-confidence policy, NOT a replacement ML probability.
    if "FALLBACK" in str(coverage_status):
        return "PASS","VERY LOW","fallback winner model + incomplete enriched coverage"
    if raw_prob < 0.60 or support < 0.67:
        return "PASS","LOW","winner edge/module support is insufficient"
    if direct_coverage == "LOW":
        if support >= 0.999 and spread <= 0.25 and raw_prob >= 0.75:
            return "LEAN","MEDIUM-HIGH","all skill modules agree but direct pre-UFC skill-stat coverage is low"
        if support >= 0.999 and raw_prob >= 0.75:
            return "LEAN","MEDIUM","all modules agree, but module dispersion/direct skill coverage prevent high confidence"
        if support >= 0.83 and raw_prob >= 0.70:
            return "LEAN","MEDIUM","most modules agree, but direct skill coverage is low and module dispersion is large"
        return "PASS","LOW","low direct skill coverage prevents trusting the raw winner probability"
    if support >= 0.999 and spread <= 0.25 and raw_prob >= 0.75:
        return "LEAN","HIGH","broad model agreement with direct skill coverage"
    if support >= 0.83 and raw_prob >= 0.65:
        return "LEAN","MEDIUM-HIGH","broad agreement"
    return "PASS","MEDIUM","mixed matchup evidence"


def main():
    payload=json.loads(LOCK_JSON.read_text())
    lock=pd.read_csv(LOCK_CSV).set_index("fight")
    mod=pd.read_csv(MOD).set_index("fight")
    tape=pd.read_csv(TAPE).set_index("fighter_name")
    details={f["fight"]:f for f in payload["fights"]}
    seeds=np.random.SeedSequence(2026100602).spawn(len(details))
    out=[]; full=[]
    for idx,(fight,d) in enumerate(details.items()):
        lr=lock.loc[fight]; mr=mod.loc[fight]
        fa,fb=fight.split(" vs ",1)
        pick=str(lr.winner_pick); pick_a=pick==fa
        raw=float(lr.winner_probability)
        support=0
        module_probs={}
        for m in MODULES:
            pa=float(mr[f"p_{m}_a"]); module_probs[m]=pa
            module_pick_a=pa>=0.5
            support+=int(module_pick_a==pick_a)
        support/=len(MODULES)
        spread=float(mr.module_spread)
        direct=str(mr.direct_skill_stat_coverage)
        status,confidence,reason=audit_policy(raw,str(lr.coverage_status),direct,support,spread)

        joint=d["joint_winner_method"]
        jp=np.array([float(joint[k]) for k in JOINT],dtype=float); jp/=jp.sum()
        rp=np.array([float(d["round"][k]) for k in ROUNDS],dtype=float); rp/=rp.sum()
        rng=np.random.default_rng(seeds[idx])
        jdraw=rng.choice(len(JOINT),size=10000,p=jp)
        rdraw=rng.choice(len(ROUNDS),size=10000,p=rp)
        jcounts={k:int(np.sum(jdraw==i)) for i,k in enumerate(JOINT)}
        rcounts={k:int(np.sum(rdraw==i)) for i,k in enumerate(ROUNDS)}

        row={
            "fight":fight,"winner_pick":pick,"raw_winner_probability":raw,
            "skill_audit_status":status,"audited_confidence":confidence,
            "module_support_fraction":support,"module_spread":spread,
            "direct_skill_stat_coverage":direct,"audit_reason":reason,
            "method_forecast":str(lr.method_forecast),"method_status":str(lr.method_status),
            "under_0_5_probability":float(lr.under_0_5_probability),
            "under_1_5_probability":float(lr.under_1_5_probability),
            "under_2_5_probability":float(lr.under_2_5_probability),
            "goes_distance_probability":float(lr.goes_distance_probability),
            "round_2_starts_probability":float(lr.round_2_starts_probability),
            "round_3_starts_probability":float(lr.round_3_starts_probability),
            "round_top":str(lr.round_top),"round_top_probability":float(lr.round_top_probability),
            "fresh_simulation_runs":10000,
        }
        out.append(row)
        full.append({
            **row,
            "module_probabilities_fighter_a":module_probs,
            "tape_evidence":{fa:str(tape.loc[fa].evidence_class),fb:str(tape.loc[fb].evidence_class)},
            "fresh_10000_joint_counts":jcounts,
            "fresh_10000_round_counts":rcounts,
            "note":"Winner probability remains the frozen production/fallback probability. The new audit changes confidence/bet status, not history. M/D/R probabilities remain the already-frozen production outputs; Monte Carlo was freshly resampled.",
        })
    pd.DataFrame(out).to_csv(DIAG/"week9_skill_audited_rerun.csv",index=False)
    report={
        "event":"DWCS Season 10 Week 9",
        "rerun_timestamp_context":"2026-10-06 pre-event-start model development",
        "winner_model":"DWCS-W-v0.6 production with existing coverage fallback",
        "new_component":"DWCS-W-SKILL-AUDIT-v0.1 prospective confidence/betting policy",
        "challenger_results":{
            "DWCS-W-v0.7-modular":"NOT PROMOTED — worse historical accuracy/Brier/log loss than full model",
            "DWCS-W-v0.8-evidence-calibrated":"NOT PROMOTED — failed historical promotion gate",
            "DWCS-W-CG-v0.1":"NOT PROMOTED — confidence calibration worse than raw anchor",
        },
        "policy":"Raw winner probabilities are retained for grading. Betting confidence is capped/downgraded when direct skill coverage is low or matchup modules disagree. No manual replacement probabilities are invented.",
        "fights":full,
    }
    (REPORTS/"week9_skill_audited_rerun.json").write_text(json.dumps(report,indent=2))

    regp=MODELS/"model_registry.json"; reg=json.loads(regp.read_text())
    for key in ["DWCS-W-v0.7-modular-challenger","DWCS-W-v0.8-evidence-calibrated-challenger","DWCS-W-CG-v0.1"]:
        if key in reg: reg[key]["status"]="NOT PROMOTED — historical validation gate failed"
    reg["DWCS-W-SKILL-AUDIT-v0.1"]={
        "status":"ACTIVE prospective audit/betting-confidence policy; not a probability model",
        "started":"2026-10-06 before Week 9 event start",
        "winner_model_remains":"DWCS-W-v0.6",
        "purpose":"Prevent resume/form-heavy probabilities from being treated as high-confidence bets when direct matchup skill evidence is weak or modular skill signals conflict.",
        "week9_rerun":"diagnostics/week9_skill_audited_rerun.csv",
        "report":"reports/week9_skill_audited_rerun.json",
    }
    regp.write_text(json.dumps(reg,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
