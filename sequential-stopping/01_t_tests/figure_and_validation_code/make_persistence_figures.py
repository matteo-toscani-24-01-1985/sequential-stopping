"""Reconstruct Figure 1 and Figure 2 from the Dropbox simulation-result tables.
Values transcribed without alteration from ASSUMED_minimum_N_P80.csv and
ASSUMED_minimum_N_P95.csv in each of the two folders indicated below.
`NA` represents the source's NaN: no interpolation or imputation.
The output is the *uncensored* assumed-true-effect persistence minima,
NOT the separately published, H0-FPR-screened PAPER_assumed_safe tables.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
SOURCE = {
    'independent': '/p-value2.0/ttest_persistence_80_95_results/',
    'paired': '/p-value2.0/paired_ttest_persistence_80_95_results/'
}

# Exact assumed_true_d column present in each of the four source CSVs.
effects = np.array([
 0, 0.0133605726778874, 0.0284245763075028, 0.0454091929720744,
 0.0645592947866616, 0.0861509742900331, 0.110495524950395,
 0.137943929173068, 0.16889191851501, 0.203785679060762,
 0.243128284215862, 0.287486947661167, 0.337501201036018,
 0.393892114250256, 0.457472691356994, 0.529159591866442,
 0.609986346490328, 0.701118257852138, 0.80386920099099,
 0.919720565876857, 1.05034261503685, 1.19761856421158,
 1.36367173321946, 1.55089615847061, 1.76199110848005, 2
])
# Columns: four ordinary p-values, four N=50-matched BFs, four reference BFs.
# Each line: zero-indexed effect row followed by the 12 source data values.
# Any source row not listed contained only NaN (all 12 entries).
RAW = {
 'independent': {
 80: '''
6 860 NA NA NA NA NA NA NA NA NA NA NA
7 560 800 NA NA 800 NA NA NA 960 NA NA NA
8 380 540 760 980 520 660 880 NA 620 760 880 NA
9 260 360 520 680 340 440 600 740 420 520 600 680
10 180 260 360 480 220 300 420 520 280 360 420 480
11 120 180 260 340 150 220 280 360 190 240 280 340
12 90 130 190 240 110 150 200 260 130 170 200 240
13 70 90 140 180 80 100 150 190 100 120 150 170
14 50 70 100 130 50 80 110 140 70 90 110 120
15 34 50 80 100 36 60 80 100 50 70 80 90
16 24 36 60 80 26 38 60 80 34 50 60 70
17 18 28 40 60 18 28 50 60 26 34 50 50
18 14 20 30 40 14 20 30 50 18 26 32 38
19 10 16 24 32 10 16 24 32 14 20 24 28
20 8 12 18 24 8 12 18 24 10 14 18 22
21 6 10 14 20 6 10 14 20 8 12 14 18
22 6 8 12 16 4 8 12 16 6 10 12 14
23 4 6 10 12 4 6 10 12 6 8 10 12
24 4 6 8 10 4 4 8 10 4 6 8 10
25 4 4 6 8 4 4 6 8 4 6 6 8
''',
 95: '''
8 740 1000 NA NA NA NA NA NA NA NA NA NA
9 500 680 940 NA 660 820 NA NA 780 940 NA NA
10 360 480 660 820 460 580 740 880 540 640 740 820
11 260 340 460 580 320 400 520 620 380 440 520 580
12 180 260 340 420 220 280 360 440 260 320 380 420
13 130 180 260 320 150 200 260 320 190 240 260 300
14 100 140 180 240 110 150 190 240 140 170 190 220
15 80 100 140 170 80 110 140 180 100 120 140 160
16 60 80 100 130 60 80 110 130 70 90 110 120
17 40 60 80 100 40 60 80 100 60 70 80 90
18 30 50 60 80 30 50 60 80 38 50 60 70
19 22 32 50 60 22 30 50 60 28 36 50 50
20 18 24 34 50 16 24 34 50 22 28 34 38
21 14 18 26 32 12 18 26 32 16 22 26 30
22 10 14 20 26 10 14 20 26 12 16 20 24
23 8 12 16 20 6 10 16 20 10 14 16 18
24 6 8 12 16 6 8 12 16 8 10 12 14
25 6 8 10 14 4 6 10 14 6 8 10 12
'''
 },
 'paired': {
 80: '''
5 720 NA NA NA NA NA NA NA NA NA NA NA
6 440 620 880 NA 640 820 NA NA 780 940 NA NA
7 280 400 580 740 380 500 680 860 500 600 680 780
8 180 260 380 500 240 320 440 560 320 380 440 520
9 120 180 260 340 160 220 300 380 220 260 300 340
10 90 130 180 240 110 150 200 260 140 180 200 240
11 60 90 130 170 70 100 140 180 100 120 140 170
12 50 70 100 120 50 70 100 130 70 90 100 120
13 30 50 70 90 32 50 80 100 50 60 80 90
14 22 34 50 70 22 36 60 70 34 50 60 70
15 18 26 38 50 16 26 38 60 24 32 40 50
16 12 20 30 38 12 20 30 40 18 24 30 36
17 10 16 22 30 8 14 22 30 14 18 22 28
18 8 12 18 24 6 12 18 24 10 14 18 22
19 6 10 14 20 4 8 14 20 8 12 14 18
20 6 8 12 16 4 8 12 16 6 10 12 14
21 4 6 10 14 4 6 10 14 6 8 10 12
22 4 6 8 12 4 6 8 12 4 6 8 10
23 4 4 6 10 4 4 6 8 4 6 6 8
24 4 4 6 10 4 4 6 8 4 6 6 8
25 4 4 6 8 4 4 6 8 4 4 6 8
''',
 95: '''
6 860 NA NA NA NA NA NA NA NA NA NA NA
7 560 760 NA NA 760 940 NA NA 920 NA NA NA
8 380 500 680 860 480 620 780 960 580 700 800 880
9 260 340 460 580 320 420 520 640 400 480 540 600
10 180 240 340 420 220 280 360 460 280 320 380 420
11 130 170 240 300 150 190 260 320 190 220 260 300
12 90 120 170 220 100 140 180 240 130 160 180 220
13 70 90 130 160 70 100 130 170 100 120 140 150
14 50 70 90 120 50 70 100 120 70 90 100 110
15 36 50 70 90 36 50 70 90 50 60 80 90
16 26 38 60 70 26 38 60 70 36 50 60 70
17 20 28 40 60 18 28 40 60 26 34 40 50
18 16 22 30 40 14 22 30 40 20 26 32 36
19 12 18 24 32 10 16 24 32 16 20 24 28
20 10 14 20 26 8 12 18 26 12 16 20 22
21 8 12 16 20 6 10 16 20 10 12 16 18
22 6 10 14 18 4 8 12 16 8 10 12 16
23 6 8 12 14 4 6 10 14 6 8 10 12
24 4 6 10 12 4 6 10 12 6 8 10 10
25 4 6 8 10 4 6 8 10 4 6 8 10
'''
 }
}

cols = ['p_0p05','p_0p01','p_0p001','p_0p0001']
cols_ind = cols + ['BF50_1p1926','BF50_4p2698','BF50_29p551','BF50_220p43'] + [f'BFref_{n}' for n in [3,10,30,100]]
cols_pair = cols + ['BF50_0p97702','BF50_3p7572','BF50_28p477','BF50_229p35'] + [f'BFref_{n}' for n in [3,10,30,100]]

# Load unaltered numeric values into arrays; all unlisted rows are all-NaN.
arrays = {}
for design in ('independent','paired'):
    cols_this = cols_ind if design == 'independent' else cols_pair
    for pct in (80,95):
        data = np.full((26,12), np.nan)
        for line in RAW[design][pct].strip().splitlines():
            pieces = line.split()
            assert len(pieces) == 13, (design,pct,line)
            i = int(pieces[0]); assert 0 <= i < 26
            assert np.isnan(data[i]).all(), (design,pct,i)
            data[i] = [float(s) if s != 'NA' else np.nan for s in pieces[1:]]
        assert np.all(np.isfinite(data[np.isfinite(data)]))
        assert np.nanmin(data) >= 4 and np.nanmax(data) <= 1000
        arrays[(design,pct)] = data
        df = pd.DataFrame(data,columns=cols_this)
        df.insert(0,'assumed_true_d',effects)
        df.to_csv(OUT / f'figure_source_{design}_P{pct}_transcribed.csv',index=False,na_rep='NaN',float_format='%.15g')

    a=arrays[(design,80)]; b=arrays[(design,95)]
    both=np.isfinite(a)&np.isfinite(b)
    assert (a[both] <= b[both]).all(), f'80 > 95 in {design}'
    print(f'{design}: P80 {int(np.isfinite(a).sum())} of 312; P95 {int(np.isfinite(b).sum())} of 312; violations 0')

mpl.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':10.5, 'axes.titlesize':12,
    'axes.labelsize':11, 'figure.facecolor':'white', 'axes.facecolor':'white',
    'savefig.facecolor':'white', 'pdf.fonttype':42, 'svg.fonttype':'none',
    'axes.spines.top':False, 'axes.spines.right':False
})
colors = ['#087e8b','#b23a48','#8a5a15','#6943a5']

for no, design in enumerate(('independent','paired'),start=1):
    fig, axes=plt.subplots(1,3,figsize=(16.8,6.55),sharey=True)
    groups=[
        ('A   Ordinary p-value thresholds', list(range(0,4)),
         ['p < .05','p < .01','p < .001','p < .0001']),
        ('B   Reference Bayes factors', list(range(8,12)),
         [r'$BF_{10}$ ≥ 3',r'$BF_{10}$ ≥ 10',r'$BF_{10}$ ≥ 30',r'$BF_{10}$ ≥ 100']),
        ('C   N = 50 p-matched Bayes factors', list(range(4,8)),
         ([r'$BF_{10}$ ≥ 1.1926',r'$BF_{10}$ ≥ 4.2698',r'$BF_{10}$ ≥ 29.551',r'$BF_{10}$ ≥ 220.434']
          if design == 'independent' else
          [r'$BF_{10}$ ≥ 0.97702',r'$BF_{10}$ ≥ 3.75716',r'$BF_{10}$ ≥ 28.4772',r'$BF_{10}$ ≥ 229.353']))
    ]
    for ax,(title,inds,labs) in zip(axes,groups):
        for j,(ix,label) in enumerate(zip(inds,labs)):
            for pct,sty,mark in [(80,'-','o'),(95,'--','s')]:
                # Matplotlib does not connect across NaN gaps and does not extrapolate.
                y=arrays[(design,pct)][:,ix]
                ax.plot(effects,y,c=colors[j],linestyle=sty,linewidth=2.05 if pct == 80 else 1.75,
                        marker=mark,markersize=3 if pct == 80 else 2.4,
                        markeredgewidth=0,zorder=3)
        ax.set_title(title,loc='left',weight='bold',pad=14)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(.085,2.13); ax.set_ylim(3.5,1130)
        ax.set_xticks([.1,.2,.5,1,2],labels=['0.1','0.2','0.5','1','2'])
        ax.set_yticks([4,10,20,50,100,200,500,1000],labels=['4','10','20','50','100','200','500','1,000'])
        ax.tick_params(axis='both',length=4.2,color='#888888',pad=5)
        ax.grid(False)
        ax.set_xlabel('Assumed true effect size (Cohen’s d)',labelpad=11)
        handles=[Line2D([0],[0],color=colors[j],lw=2.4,linestyle='-',label=labs[j]) for j in range(4)]
        ax.legend(handles=handles,loc='upper right',fontsize=9,frameon=False,ncol=1,
                  bbox_to_anchor=(1.01,1.0),handlelength=2.2,borderpad=0.1,labelspacing=0.55)
    axes[0].set_ylabel(('Minimum participants per group (n)' if design == 'independent' else
                        'Minimum number of pairs (n)'),labelpad=11)
    fig.suptitle((f'Figure {no}. ' + ('Independent-samples t test' if design == 'independent' else 'Paired-samples t test') +
                 ' — evidence persistence'),fontsize=16,weight='bold',x=.06,y=.985,ha='left')
    style_handles=[Line2D([0],[0],color='#272727',lw=2.2,linestyle='-',marker='o',markersize=4,label='80% persistence'),
                   Line2D([0],[0],color='#272727',lw=2.2,linestyle='--',marker='s',markersize=4,label='95% persistence')]
    fig.legend(handles=style_handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=2,
               frameon=False,fontsize=11,handlelength=3.6,columnspacing=5)
    fig.text(.065,.11,'Lower 95% Wilson confidence bound used for each persistence minimum; 1,000 subsequent looks.\n'
                  'Raw checkpoint-calibrated minima (not FPR-screened). Gaps represent unavailable/ineligible source cells.',
             ha='left',fontsize=9,color='#51545a',linespacing=1.35)
    fig.subplots_adjust(left=.07,right=.985,top=.87,bottom=.235,wspace=.15)
    stem = f'Figure_{no}_{design}_ttest_80_95_persistence_3panel'
    for ext in ['png','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.22)
    plt.close(fig)
    print('Saved',stem)

(OUT/'FIGURES_README.txt').write_text('''Figure 1: independent-samples t test. Figure 2: paired-samples t test.
Each comprises 3 panels (p-value thresholds, reference Bayes factors,
p-matched Bayes factors at N=50), and draws both 80% and 95% curves
for each of the 12 thresholds, using original simulation CSV values.

Dropbox original source files:
/p-value2.0/ttest_persistence_80_95_results/ASSUMED_minimum_N_P80.csv
/p-value2.0/ttest_persistence_80_95_results/ASSUMED_minimum_N_P95.csv
/p-value2.0/paired_ttest_persistence_80_95_results/ASSUMED_minimum_N_P80.csv
/p-value2.0/paired_ttest_persistence_80_95_results/ASSUMED_minimum_N_P95.csv

The 4 figure_source_*.csv files are numeric transcriptions of those originals
made from Dropbox extracted content for local plotting; they are not claims
of a direct raw-byte Dropbox download. No new simulations were run.
All missing NaN source values stay missing; no interpolation or imputation.
Matplotlib connects adjacent existing grid points only, never over missing cells.

Persistence definition: threshold crossed at checkpoint n and remains above
threshold during all next 1,000 observations. The minimum is selected on a
discrete N grid where the one-sided 95% Wilson lower bound for persistence
exceeds the target, with minimum 200 qualifying runs.
Plots are the original unscreened checkpoint minima, not the separate
PAPER_assumed_safe tables filtered on the held-out null stopping false-positive
upper bound of less than 0.05. Do not describe these as FPR-controlled plots.
The 80% and 95% curves can coincide at some points.

Independent sample size is per group (2n participants total).
Paired sample size is number of pairs/participants.
''')
