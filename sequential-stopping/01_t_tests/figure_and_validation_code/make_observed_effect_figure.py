"""Reproduce a composite figure of observed-effect checkpoint persistence minima.
Data in DATA are transcribed verbatim from the four source Dropbox CSVs:
/p-value2.0/{ttest_persistence_80_95_results,paired_ttest_persistence_80_95_results}/OBSERVED_minimum_N_{P80,P95}.csv
These are RAW checkpoint minima, NOT false-positive-rate screened safe stopping tables.
"""
from pathlib import Path
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
HEAD_I = 'p_0p05,p_0p01,p_0p001,p_0p0001,BF50_1p1926,BF50_4p2698,BF50_29p551,BF50_220p43,BFref_3,BFref_10,BFref_30,BFref_100'
HEAD_P = 'p_0p05,p_0p01,p_0p001,p_0p0001,BF50_0p97702,BF50_3p7572,BF50_28p477,BF50_229p35,BFref_3,BFref_10,BFref_30,BFref_100'
BINS = [(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1),(1,1.2),(1.2,1.4),(1.4,1.6),(1.6,1.8),(1.8,2),(2,2.25),(2.25,2.5),(2.5,3),(3,4),(4,6),(6,np.inf)]
# Four complete 21 x 12 matrices, aligned to observed-effect bins in BINS.
DATA = {}
DATA['independent',80] = '''
NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
960,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
280,360,460,560,340,440,500,600,400,480,500,560
130,180,260,320,150,190,280,340,180,240,280,320
80,110,150,200,90,110,160,220,100,130,160,190
50,70,100,130,60,70,100,140,70,90,100,120
36,50,70,100,36,50,70,100,50,60,70,90
26,36,60,70,26,36,60,70,34,50,60,70
22,28,50,60,20,28,50,60,26,34,50,50
18,24,34,50,16,22,34,50,20,28,34,40
14,18,26,34,12,18,26,34,16,22,26,32
10,14,20,26,10,14,20,26,12,16,20,24
8,12,16,20,8,10,16,20,10,12,16,18
8,10,14,18,6,10,14,18,8,10,14,16
6,8,12,14,6,8,12,14,8,10,12,14
6,8,10,14,6,8,10,14,6,8,10,12
6,8,10,12,6,6,10,12,6,8,10,10
6,6,8,10,4,6,8,10,6,8,8,10
4,6,8,10,4,6,8,10,6,6,8,8
4,6,6,8,4,6,6,8,4,6,6,8
4,6,6,NaN,4,4,6,NaN,4,6,6,NaN
'''
DATA['independent',95] = '''
NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
400,480,640,840,460,560,740,920,540,640,740,840
190,240,300,400,220,260,340,420,240,300,340,380
110,140,180,240,120,140,190,240,140,160,190,220
70,90,120,150,80,100,120,160,90,110,120,140
50,70,90,110,60,70,90,110,60,80,90,100
38,50,70,80,38,50,70,90,50,60,70,80
30,38,50,70,30,38,50,70,36,50,50,60
24,30,40,60,24,30,40,60,28,34,40,50
18,24,32,40,18,24,32,40,22,28,32,38
14,18,24,30,14,18,24,30,16,20,24,28
12,14,18,24,12,14,18,24,14,16,18,22
10,12,16,20,10,12,16,20,12,14,16,18
8,10,14,18,8,10,14,18,10,12,14,16
8,10,12,16,8,10,12,16,8,10,12,14
8,8,12,14,8,8,12,14,8,10,12,12
6,8,10,12,6,8,10,12,8,10,10,12
6,8,10,12,6,8,10,12,6,8,10,10
6,6,8,10,6,6,8,10,6,8,8,10
6,6,NaN,NaN,6,6,NaN,NaN,6,NaN,NaN,NaN
'''
DATA['paired',80] = '''
NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
500,560,540,640,640,540,600,700,540,560,600,660
140,190,240,280,160,220,260,300,200,240,260,280
70,90,130,160,80,100,140,170,90,120,140,160
40,60,80,100,50,60,80,110,60,70,80,100
28,36,60,70,26,36,60,80,34,50,60,70
20,26,38,60,20,26,38,60,24,32,40,50
16,22,30,40,14,20,30,40,20,24,30,36
12,18,24,32,12,16,24,32,16,20,24,30
10,14,20,28,10,14,20,28,12,16,20,24
8,12,16,22,8,10,16,20,10,14,16,20
8,10,14,18,6,8,14,16,8,10,14,16
6,8,12,14,6,8,12,14,8,10,12,12
6,8,10,14,4,6,10,12,6,8,10,12
6,8,10,12,4,6,10,12,6,8,10,10
6,6,8,12,4,6,8,12,6,8,8,10
6,6,8,10,4,6,8,10,6,6,8,10
4,6,8,10,4,6,8,10,6,6,8,8
4,6,8,10,4,6,8,10,4,6,8,8
4,6,8,8,4,6,6,8,4,6,6,8
4,6,6,8,4,4,6,8,4,6,6,8
'''
DATA['paired',95] = '''
NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
720,860,NaN,NaN,880,NaN,NaN,NaN,NaN,NaN,NaN,NaN
200,240,340,420,240,280,380,480,280,320,380,440
100,120,160,200,110,130,170,220,120,150,170,200
60,70,100,120,60,80,100,130,70,90,100,120
38,50,70,90,40,50,70,90,50,60,70,80
28,36,50,60,28,36,50,60,34,50,50,60
22,28,38,50,22,28,36,50,26,32,38,50
18,22,30,38,18,22,30,38,22,26,30,34
16,18,26,32,14,18,24,32,18,22,26,30
12,16,20,26,12,14,20,26,14,18,20,24
10,12,16,22,10,12,16,20,12,14,16,18
8,10,14,18,8,10,14,18,10,12,14,16
8,10,12,16,8,10,12,16,8,10,12,14
8,8,12,14,6,8,12,14,8,10,12,12
8,8,10,14,6,8,10,14,8,10,10,12
6,8,10,12,6,8,10,12,8,8,10,12
6,8,10,12,6,8,10,12,6,8,10,10
6,8,10,12,6,6,8,10,6,8,8,10
6,6,8,10,6,6,8,10,6,8,8,10
6,6,8,10,6,6,8,10,6,8,8,10
'''

def df_for(design, target):
    s=DATA[design,target]
    cols=(HEAD_I if design=='independent' else HEAD_P).split(',')
    table=pd.read_csv(StringIO(s.strip()),header=None,names=cols)
    assert table.shape==(21,12),table.shape
    for col in table:
        bad=table[col].dropna()
        assert (bad>=4).all() and (bad<=1000).all()
    table.insert(0,'observed_abs_d_bin',[f'[{lo:.2f}, {hi:.2f})' if np.isfinite(hi) else f'[{lo:.2f}, Inf)' for lo,hi in BINS])
    return table

files=[]
for d in ['independent','paired']:
 for t in [80,95]:
    tab=df_for(d,t)
    outfile=ROOT/f'Figure_3_source_{d}_{t}pct.csv'
    tab.to_csv(outfile,index=False,na_rep='NaN')
    files.append(outfile)
    print(outfile.name, 'nonmissing:',tab.iloc[:,1:].notna().sum().sum())

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12.5,'axes.labelsize':11,'legend.fontsize':9,'svg.fonttype':'none'})
fig, axs=plt.subplots(2,3,figsize=(17.5,11.4),sharex=True,sharey=True)
colors=['#087E91','#B73748','#8F570B','#6240A5']
xs=np.array([b[0] for b in BINS])
p_labels=['p < .05','p < .01','p < .001','p < .0001']
ref_labels=[r'$BF_{10}\geq 3$',r'$BF_{10}\geq 10$',r'$BF_{10}\geq 30$',r'$BF_{10}\geq 100$']
labels_pmatch_i=[r'$BF_{10}\geq 1.1926$',r'$BF_{10}\geq 4.2698$',r'$BF_{10}\geq 29.551$',r'$BF_{10}\geq 220.434$']
labels_pmatch_p=[r'$BF_{10}\geq 0.9770$',r'$BF_{10}\geq 3.7572$',r'$BF_{10}\geq 28.477$',r'$BF_{10}\geq 229.353$']

for i,design in enumerate(['independent','paired']):
    data80=df_for(design,80)
    data95=df_for(design,95)
    for j,(indices,title) in enumerate([(range(0,4),'Ordinary p-value thresholds'),(range(8,12),'Reference Bayes factors'),(range(4,8),'N = 50 p-matched Bayes factors')]):
        ax=axs[i,j]
        for n,colidx in enumerate(indices):
            c=data80.columns[colidx+1]
            y80=data80[c].to_numpy(dtype=float)
            y95=data95[c].to_numpy(dtype=float)
            # log-scale requires x >0; [0,.10) is all unavailable, omitted.
            ax.plot(xs[1:],y80[1:],color=colors[n],lw=1.7,ls='-',marker='o',markersize=2.0)
            ax.plot(xs[1:],y95[1:],color=colors[n],lw=1.7,ls='--',marker='s',markersize=1.8)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.095,6.5); ax.set_ylim(3.5,1200)
        ax.set_xticks([.1,.2,.3,.5,1,2,4,6])
        ax.set_xticklabels(['.1','.2','.3','.5','1','2','4','6'])
        ax.set_yticks([4,10,20,50,100,200,500,1000])
        ax.set_yticklabels(['4','10','20','50','100','200','500','1,000'])
        ax.tick_params(axis='both',which='major',length=4,direction='out')
        ax.minorticks_off()
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(chr(65+i*3+j)+'   '+title,loc='left',weight='bold',pad=12)
        labs=[p_labels,ref_labels,labels_pmatch_i if design=='independent' else labels_pmatch_p][j]
        handles=[Line2D([0],[0],color=colors[k],lw=2.5) for k in range(4)]
        ax.legend(handles,labs,loc='upper right',frameon=False,ncol=1,fontsize=9.4,handlelength=2.2)
        ax.set_xlabel('Observed |d| (bin lower bound)')
        ax.tick_params(axis='x',labelbottom=True)
        if j==0: ax.set_ylabel('Minimum N (participants per group)' if i==0 else 'Minimum N (paired participants)')

fig.text(.055,.92,'INDEPENDENT-SAMPLES t TEST',fontsize=13.7,fontweight='bold')
fig.text(.055,.535,'PAIRED-SAMPLES t TEST',fontsize=13.7,fontweight='bold')
fig.suptitle('Figure 3. Observed-effect persistence calibration — all 12 evidence thresholds',fontsize=20.5,fontweight='bold',x=.05,y=.994,ha='left')
fig.subplots_adjust(left=.07,right=.986,top=.87,bottom=.185,wspace=.17,hspace=.47)
style=[Line2D([0],[0],color='#222222',lw=2.4,marker='o',markersize=3,ls='-'),Line2D([0],[0],color='#222222',lw=2.4,marker='s',markersize=3,ls='--')]
fig.legend(style,['80% checkpoint persistence','95% checkpoint persistence'],loc='lower center',bbox_to_anchor=(.5,.099),ncol=2,frameon=False,fontsize=11.8)
fig.text(.05,.038,'Lower one-sided 95% Wilson bound used; 1,000 consecutive additional looks. Raw checkpoint minima, not false-positive-screened.',fontsize=10.0)
fig.text(.05,.019,'Unavailable minima remain missing; plotted x values are observed-effect-bin lower boundaries (6 denotes the open-ended bin).',fontsize=10.0)
base=ROOT/'Figure_3_observed_effect_ttests_80_95_all_thresholds'
fig.savefig(str(base)+'.png',dpi=200,facecolor='white',bbox_inches='tight')
fig.savefig(str(base)+'.svg',facecolor='white',bbox_inches='tight')
plt.close(fig)
print('OUTPUT',base)
