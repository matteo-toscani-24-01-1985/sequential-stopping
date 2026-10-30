#!/usr/bin/env python3
"""Redraw the manuscript's Figure 13 from saved simulation results, without rerunning simulations."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent
r=pd.read_csv(root/'results'/'Figure_13_robustness_all12_source_data.csv')
order=[('t_ind','Independent t-test'),('t_pair','Paired t-test'),('one_ind','Independent one-way ANOVA'),('one_rep','Repeated one-way ANOVA'),('fact2_ind','Independent 2 x 2 interaction'),('fact2_rep','Repeated 2 x 2 interaction'),('fact3_ind','Independent 2 x 2 x 2 interaction'),('fact3_rep','Repeated 2 x 2 x 2 interaction')]
conditions=[('normal','Normal'),('t3','Student t(3)'),('lognormal','Skewed lognormal'),('variance9','Balanced variances 9:1'),('sphericity55','Sphericity epsilon=.55')]
labels=['p < .05','p < .01','p < .001','p < .0001','BF matched .05','BF matched .01','BF matched .001','BF matched .0001','BF >= 3','BF >= 10','BF >= 30','BF >= 100']
fig,axes=plt.subplots(4,2,figsize=(17,18),sharex=True,sharey=True)
colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
for ax,(key,title) in zip(axes.flat,order):
    sub=r[r['code']==key]
    ax.axhline(5,color='firebrick',ls='--',lw=1.3)
    for j,(condition,caption) in enumerate(conditions):
        d=sub[sub['distribution']==condition].sort_values('criterion_index')
        if len(d):ax.plot(d['criterion_index'],d['FPR_pct'],marker='o',markersize=4,lw=1.6,color=colors[j],label=caption)
    ax.set_title(title,fontsize=12)
    ax.set_ylim(0,max(30,min(65,r.FPR_pct.max()+3)))
    ax.grid(alpha=.15)
for ax in axes[-1]:
    ax.set_xticks(range(12),labels,rotation=65,ha='right',fontsize=8)
for ax in axes[:,0]:ax.set_ylabel('Sequential null false-positive rate (%)')
fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=5,bbox_to_anchor=(.5,.995),frameon=False)
fig.suptitle('Assumption checks across all 12 evidence thresholds | 95% minimum-N rules',y=1.01,fontsize=15)
fig.tight_layout(rect=[0,0,1,.97])
output=root/'figures'/'Figure_13_redrawn_from_saved_data.png'
fig.savefig(output,dpi=180,bbox_inches='tight')
print('Created:',output)
