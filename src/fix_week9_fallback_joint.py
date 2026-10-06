from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/week9_preodds_model_lock.json'
CSV=ROOT/'official_picks/week9_preodds_model_lock.csv'
METHOD3=['KO_TKO','SUB','DEC']
JOINT=['A_KO_TKO','A_SUB','A_DEC','B_KO_TKO','B_SUB','B_DEC']

def main():
    payload=json.loads(REPORT.read_text())
    table=pd.read_csv(CSV)
    seeds=np.random.SeedSequence(2026100602).spawn(len(payload['fights']))
    for i,f in enumerate(payload['fights']):
        if f['coverage_status']!='V0_3_FALLBACK':
            continue
        a,b=f['fight'].split(' vs ',1)
        p_a=float(f['winner_probabilities']['v0_3_fallback_a'])
        pa=np.array([f['conditional_method_if_a_wins'][m] for m in METHOD3],dtype=float)
        pb=np.array([f['conditional_method_if_b_wins'][m] for m in METHOD3],dtype=float)
        joint=np.r_[p_a*pa,(1-p_a)*pb]
        joint=joint/joint.sum()
        pick_a=f['winner_pick']==a
        side=joint[:3] if pick_a else joint[3:]
        cond=pa if pick_a else pb
        j=int(np.argmax(side)); method=METHOD3[j]
        f['method_forecast']=f"{f['winner_pick']} by {method}"
        f['method_joint_probability']=float(side[j])
        f['method_conditional_probability_given_win']=float(cond[j])
        f['joint_winner_method']=dict(zip(JOINT,map(float,joint)))
        rng=np.random.default_rng(seeds[i])
        draw=rng.choice(6,size=10000,p=joint)
        f['monte_carlo']['joint_outcome_counts_10000']={JOINT[k]:int((draw==k).sum()) for k in range(6)}
        f['monte_carlo']['note']='Fallback-gated fights use v0.3 winner probability consistently with M-v0.4 conditional method probabilities. Primary fights use W-v0.6. D/R-v0.4 timing outputs are unchanged.'
        mask=table.fight.eq(f['fight'])
        table.loc[mask,'method_forecast']=f['method_forecast']
        table.loc[mask,'method_joint_probability']=f['method_joint_probability']
        table.loc[mask,'method_conditional_probability_given_win']=f['method_conditional_probability_given_win']
    payload['coherence_audit']='Fallback-gated fights use the same v0.3 winner component for side, joint winner+method probabilities, and Monte Carlo. Primary-coverage fights use W-v0.6.'
    REPORT.write_text(json.dumps(payload,indent=2))
    table.to_csv(CSV,index=False)
    print(table.to_string(index=False))

if __name__=='__main__':
    main()
