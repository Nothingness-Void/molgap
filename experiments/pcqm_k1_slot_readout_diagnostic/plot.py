"""Render a standalone scientific summary from accepted observations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
j=json.loads((HERE/'results/analysis.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
colors=['#3569a8','#d17734']
x=np.arange(3)
for arm,label,color,shift in [('reference','192',colors[0],-.18),('candidate','256',colors[1],.18)]:
    layers=j['arms'][arm]['layers']
    med=np.array([layers[str(l)]['update_to_pre_mixer_atom_mean_norm_ratio']['median'] for l in [3,6,9]])
    axes[0].bar(x+shift,med*100,.34,label=label,color=color)
    for a,b in zip(x+shift,med*100):axes[0].text(a,b+2,f'{b:.1f}%',ha='center',fontsize=9)
axes[0].set(xticks=x,xticklabels=['Layer 3','Layer 6','Layer 9'],ylim=(0,110),
    ylabel='Median ||mean(update)|| / ||mean(hidden)|| (%)',title='Slot update remains substantial')
axes[0].legend(title='Atom width')
cohorts=j['full50k_structure']['cohorts']['conjugated_bond_fraction']
gain=np.array([c['reference_minus_candidate_mean_AE_eV']*1000 for c in cohorts])
ci=np.array([c['paired_row_bootstrap_95pct_eV'] for c in cohorts]).T*1000
axes[1].bar(np.arange(4),gain,color=[colors[1] if v<0 else colors[0] for v in gain])
axes[1].errorbar(np.arange(4),gain,yerr=np.stack([gain-ci[0],ci[1]-gain]),fmt='none',color='#333',capsize=4)
axes[1].axhline(0,color='#777',linewidth=.8)
axes[1].set(xticks=np.arange(4),xticklabels=[f"Q{i+1}\nn={c['n']:,}" for i,c in enumerate(cohorts)],
    ylabel='192 MAE - 256 MAE (meV); positive favors 256',
    title='Highest conjugation quartile regresses more')
fig.suptitle('Frozen K1 checkpoints: 2,048 forwards per arm; 50,000 retained development rows',fontsize=11)
fig.savefig(HERE/'results/summary.png',dpi=180)
