"""Generate five ANOVA Results figures from validated Dropbox CSV transcriptions.
Default 3-level independent and repeated-measures one-way ANOVA.
Figure 5 displays overall observed-effect stopping-rule FPR by threshold;
per-bin null FPRs were not exported by the ANOVA simulation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
fgrid=np.array([0,.00668028633894368,.0142122881537514,.0227045964860372,.0322796473933308,.0430754871450165,.0552477624751977,.0689719645865342,.0844459592575052,.101892839530381,.121564142107931,.143743473830584,.168750600518009,.196946057125128,.228736345678497,.264579795933221,.304993173245164,.350559128926069,.401934600495495,.459860282938428,.525171307518423,.598809282105789,.681835866609728,.775448079235306,.880995554240026,1.0])
bin_names=['[0.00, 0.05)','[0.05, 0.10)','[0.10, 0.15)','[0.15, 0.20)','[0.20, 0.25)','[0.25, 0.30)','[0.30, 0.35)','[0.35, 0.40)','[0.40, 0.45)','[0.45, 0.50)','[0.50, 0.60)','[0.60, 0.70)','[0.70, 0.80)','[0.80, 0.90)','[0.90, 1.00)','[1.00, 1.20)','[1.20, 1.50)','[1.50, 2.00)','[2.00, 3.00)','[3.00, Inf)']
bins=np.array([0,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.60,.70,.80,.90,1.,1.2,1.5,2.,3.])

# Grid rows in original simulation order. Dash means no calibrated minimum N.
N={}
def addN(name,target,kind,raw):
    rows=[]
    for ln in raw.strip().splitlines():
        row=[np.nan if x=='-' else float(x) for x in ln.split()]
        assert len(row)==12,(name,target,kind,len(row),ln)
        rows.append(row)
    a=np.array(rows,dtype=float)
    if kind=='assumed':
        assert a.shape==(20,12)
        a=np.vstack([np.full((6,12),np.nan),a])
    else: assert a.shape==(20,12)
    N[(name,target,kind)]=a

addN('ind',80,'assumed', '''
740 1000 - - - - - - - - - -
460 640 880 - 740 900 - - 880 1000 - -
300 420 580 740 460 580 720 880 540 640 720 820
220 300 400 500 300 380 480 580 360 420 480 540
150 200 280 360 200 260 320 400 240 280 320 380
110 140 200 260 130 170 220 280 160 190 220 260
80 100 140 180 90 120 160 200 110 140 160 180
60 80 110 130 60 80 110 140 80 100 110 130
38 60 80 100 50 60 80 100 60 70 80 90
28 40 60 70 30 50 60 80 38 50 60 70
22 30 50 60 20 30 50 60 28 36 50 50
16 22 32 40 14 22 32 40 20 26 32 36
12 16 24 30 10 16 24 30 16 20 24 28
10 12 18 24 8 12 18 24 12 14 18 22
8 10 14 18 6 10 14 18 8 12 14 16
6 8 12 14 4 8 12 14 6 10 12 12
4 6 10 12 4 6 8 12 6 8 8 10
4 6 8 10 4 6 8 10 4 6 8 8
4 4 6 8 4 4 6 8 4 6 6 8
4 4 6 6 4 4 6 6 4 4 6 6''')
addN('ind',95,'assumed','''
- - - - - - - - - - - -
900 - - - - - - - - - - -
600 780 - - 860 - - - 1000 - - -
400 540 700 860 580 680 840 1000 660 760 840 920
280 380 500 600 380 460 580 680 440 520 580 640
200 280 360 440 260 320 400 480 300 360 400 440
150 190 260 320 180 220 280 340 220 260 280 320
110 140 190 240 130 160 200 240 150 180 200 240
80 100 140 170 90 110 150 180 110 130 150 170
60 80 100 130 60 80 110 130 80 90 110 120
50 60 80 100 50 60 80 100 60 70 80 90
32 50 60 70 32 50 60 70 40 50 60 70
24 32 50 60 22 32 50 60 30 36 50 50
18 24 32 40 16 24 32 40 22 28 32 38
14 18 26 32 12 18 24 32 16 22 24 28
10 14 20 24 10 14 20 24 12 16 20 22
8 12 16 18 8 10 14 18 10 12 14 18
6 8 12 16 6 8 12 16 8 10 12 14
6 8 10 12 4 6 10 12 6 8 10 10
4 6 8 10 4 6 8 10 6 6 8 10''')
addN('rm',80,'assumed','''
720 1000 - - - - - - - - - -
460 640 880 - 800 980 - - 980 - - -
320 420 580 740 500 620 780 940 620 720 800 880
220 300 400 520 320 400 520 620 420 480 540 600
150 200 280 360 220 280 360 440 280 320 360 400
110 140 200 260 140 180 240 300 190 220 260 280
80 110 150 180 100 130 170 220 130 150 170 200
60 80 110 140 70 90 120 150 90 110 120 140
38 60 80 100 50 70 90 110 70 80 90 100
28 40 60 80 30 50 70 80 50 60 70 80
22 30 50 60 22 32 50 60 32 40 50 60
16 22 32 50 14 22 34 50 24 30 34 40
12 18 24 32 10 16 24 32 18 22 26 30
10 14 20 24 8 12 18 24 12 16 20 22
8 10 16 20 6 10 14 20 10 12 16 18
6 8 12 16 4 8 12 16 8 10 12 14
6 8 10 12 4 6 10 12 6 8 10 12
4 6 8 10 4 4 8 10 6 6 8 10
4 6 8 8 4 4 6 8 4 6 6 8
4 4 6 8 4 4 6 8 4 4 6 6''')
addN('rm',95,'assumed','''
- - - - - - - - - - - -
900 - - - - - - - - - - -
600 780 - - 920 - - - - - - -
400 540 700 880 600 720 900 - 740 820 920 1000
280 380 500 620 400 500 600 720 500 560 620 680
200 280 360 440 280 340 420 500 340 400 440 480
150 190 260 320 190 240 300 360 240 280 300 340
110 140 190 240 130 170 220 260 170 200 220 240
80 110 140 170 90 120 150 190 120 140 160 180
60 80 110 130 70 90 110 140 90 100 120 130
50 60 80 100 50 60 80 100 70 80 90 100
32 50 60 80 32 50 60 80 50 60 70 70
24 32 50 60 22 32 50 60 34 40 50 60
18 26 34 50 16 24 34 50 24 30 34 40
14 20 26 32 12 18 26 32 18 22 26 30
12 16 20 26 8 14 20 26 14 18 20 24
8 12 16 20 6 10 16 20 10 14 16 18
8 10 14 16 4 8 12 16 8 10 12 14
6 8 10 14 4 6 10 12 6 8 10 12
4 6 8 12 4 6 8 10 6 8 8 10''')

addN('ind',80,'observed','''
- - - - - - - - - - - -
940 880 840 980 900 880 980 - 860 900 980 -
260 320 380 440 320 380 420 480 380 380 420 440
130 150 220 240 140 170 220 260 160 200 220 240
70 90 120 150 80 100 130 160 90 110 130 150
50 60 80 110 50 60 90 110 60 70 90 100
34 50 60 80 34 50 60 80 40 50 60 70
26 32 50 60 24 32 50 60 30 36 50 50
20 26 34 50 20 24 34 50 24 28 34 40
16 20 28 36 16 20 28 36 18 24 28 32
12 16 22 26 12 16 22 26 14 18 22 24
10 12 16 20 8 12 16 20 10 14 16 18
8 10 12 16 6 10 12 16 8 10 12 14
6 8 10 14 6 8 10 14 8 10 10 12
6 8 10 12 6 6 10 12 6 8 10 10
6 6 8 10 4 6 8 10 6 8 8 10
4 6 8 8 4 6 8 8 6 6 8 8
4 6 6 8 4 4 6 8 4 6 6 6
4 4 6 6 4 4 6 6 4 4 6 6
4 4 - - 4 4 - - 4 4 - -''')
addN('ind',95,'observed','''
- - - - - - - - - - - -
- - - - - - - - - - - -
360 420 540 680 420 500 620 760 480 560 620 700
170 190 260 320 190 220 280 340 220 240 280 320
100 120 150 180 110 120 160 190 120 140 150 170
70 80 100 120 70 80 100 120 80 90 100 110
50 60 70 90 50 60 70 90 60 60 70 80
34 40 60 70 34 40 60 70 38 50 60 60
26 32 40 50 26 32 40 50 30 36 40 50
22 26 32 40 22 26 32 40 24 28 32 38
16 20 26 32 16 20 26 32 18 22 26 30
12 16 20 24 12 14 20 24 14 16 20 22
10 12 16 18 10 12 16 18 12 14 16 18
8 10 12 16 8 10 12 16 10 12 12 14
8 8 12 14 8 8 12 14 8 10 12 12
6 8 10 12 6 8 10 12 8 8 10 10
6 6 8 10 6 6 8 10 6 8 8 10
6 6 8 8 6 6 8 8 6 6 8 8
6 6 6 8 4 6 6 8 6 6 6 8
4 - - - 4 - - - - - - -''')
addN('rm',80,'observed','''
- - - - - - - - - - - -
940 860 840 980 880 900 - - 900 960 - -
260 320 380 440 340 380 440 500 380 420 440 480
130 160 220 240 150 180 240 260 190 220 240 260
80 90 130 150 80 100 140 170 110 120 140 160
50 60 90 110 50 70 90 110 70 80 90 110
34 50 60 80 34 50 60 80 50 60 70 80
26 32 50 60 26 32 50 60 32 40 50 60
20 26 36 50 20 24 34 50 26 30 36 50
16 22 28 36 16 20 28 36 20 24 30 34
14 16 22 28 12 16 22 28 16 20 22 26
10 12 18 22 8 12 16 20 12 14 16 20
8 10 14 18 8 10 12 16 10 12 14 16
8 8 12 14 6 8 10 14 8 10 12 14
6 8 10 14 6 8 10 12 8 8 10 12
6 8 10 12 4 6 8 10 6 8 8 10
6 6 8 10 4 6 8 10 6 6 8 8
4 6 8 8 4 6 6 8 6 6 6 8
4 6 6 8 4 4 6 8 4 6 6 8
4 6 6 - 4 4 6 - 4 6 6 6''')
addN('rm',95,'observed','''
- - - - - - - - - - - -
- - - - - - - - - - - -
360 420 540 680 440 540 660 800 540 620 680 760
170 200 260 320 190 240 300 360 240 280 300 340
100 120 150 180 110 130 160 200 130 150 170 190
70 80 100 120 70 80 110 130 90 100 110 120
50 60 70 90 50 60 80 90 60 70 80 90
34 40 60 70 36 50 60 70 50 50 60 70
28 32 50 60 28 32 50 60 32 38 50 50
22 26 34 50 22 26 34 50 26 30 34 40
18 22 28 34 16 20 26 34 20 24 28 32
14 16 20 26 12 16 20 24 16 18 20 24
10 12 16 20 10 12 16 20 12 14 16 18
10 12 14 18 8 10 14 16 10 12 14 16
8 10 12 14 8 10 12 14 10 10 12 14
8 8 10 14 6 8 10 12 8 10 10 12
6 8 10 12 6 8 10 12 8 8 10 10
6 8 8 10 6 6 8 10 6 8 8 10
6 6 8 10 6 6 8 10 6 8 8 8
6 6 - - 6 6 - - 6 - - -''')

# Null overall FPR for observed-effect adaptive procedure (12 criteria each).
obsFPR={
 ('ind',80):np.array([.13822875,.0557875,.0093025,.0011675,.12557125,.03806375,.0056175,.000745,.05544125,.01751875,.00565,.00203875]),
 ('ind',95):np.array([.03011125,.010855,.00158625,.00019375,.0245825,.008795,.00116,.00015,.01084375,.00373875,.00124,.000385]),
 ('rm',80): np.array([.13762625,.06032375,.008645,.0011625,.13594875,.0396425,.00726,.0009125,.0365525,.01413125,.00536375,.00134375]),
 ('rm',95): np.array([.029455,.0111375,.00171375,.00022,.03098,.0088775,.001135,.0001825,.008555,.0029975,.0009925,.0002725]),
}
naive={
'ind':np.array([.4519325,.139075,.0195875,.00238125,.458895,.11411375,.01427,.001715,.16095125,.04633375,.0146725,.00416875]),
'rm': np.array([.45566125,.14096625,.0198825,.00239875,.592335,.15592,.0181875,.00195,.135815,.04127625,.01354375,.00385625])}

# Assumed-effect null rates: each row contains the 12 Monte Carlo null stop counts
# for one (f, target). The validation used 800,000 independent null trajectories.
counts={}
def addCounts(des,target,raw):
 rows=[]
 for line in raw.strip().splitlines():
    v=[int(s) for s in line.split()]
    assert len(v)==12,(des,target,len(v),line)
    rows.append(v)
 assert len(rows)==20,(des,target,len(rows))
 counts[(des,target)]=np.vstack([np.zeros((6,12),dtype=int),np.asarray(rows,dtype=int)])
addCounts('ind',80,'''
82495 7917 0 0 0 0 0 0 0 0 0 0
123460 22946 1551 0 11723 1878 0 0 2804 425 0 0
153358 32852 3114 252 23610 4868 504 44 7415 1838 518 121
172643 39848 4381 412 36737 8383 964 102 12066 3388 992 256
197061 47606 5406 520 51935 12153 1548 166 17611 5053 1589 385
215117 54984 6270 639 73896 17398 2136 227 24816 6967 2193 563
231644 61165 7390 770 94312 22730 2794 302 32768 9082 2870 772
245241 64861 8051 899 118952 29252 3636 415 39837 11387 3738 1009
266238 69292 8821 980 130333 34065 4350 508 46385 13910 4478 1260
284311 74700 9477 1092 175820 37038 5008 566 58022 16186 5150 1425
296666 81202 9853 1133 215508 49034 5394 653 69768 19054 5545 1624
311638 87061 10984 1233 250446 56934 6645 755 81664 22699 6836 1890
323945 92483 11853 1349 282573 64583 7615 873 89325 25395 7825 2158
331121 97047 12635 1439 303488 71125 8470 962 98670 28647 8710 2368
339483 99759 13220 1545 330309 75017 9102 1065 110864 29991 9364 2646
349159 102867 13595 1630 367116 79544 9497 1147 118743 31497 9772 2826
361546 106528 14006 1675 367116 84747 10398 1189 118743 33121 10687 2955
361546 106528 14495 1727 367116 84747 10398 1237 128761 34954 10687 3084
361546 111260 15051 1791 367116 91291 10911 1289 128761 34954 11215 3084
361546 111260 15051 1854 367116 91291 10911 1337 128761 37067 11215 3222''')
addCounts('ind',95,'''
0 0 0 0 0 0 0 0 0 0 0 0
59628 0 0 0 0 0 0 0 0 0 0 0
101988 17606 0 0 8179 0 0 0 1396 0 0 0
133752 27162 2406 182 17497 3735 346 24 5448 1306 352 85
157824 35023 3643 336 29200 6739 727 75 9710 2550 748 194
178106 41203 4698 456 41841 9987 1223 135 14421 3981 1256 331
197061 48752 5611 565 57235 13904 1760 185 18879 5366 1804 460
215117 54984 6436 668 73896 18287 2293 257 26148 7368 2350 596
231644 61165 7390 795 94312 24056 2932 331 32768 9542 3012 814
245241 64861 8294 899 118952 29252 3636 433 39837 12126 3738 1064
253028 69292 8821 980 130333 34065 4350 508 46385 13910 4478 1260
276973 71803 9477 1092 169527 37038 5008 614 55686 16186 5150 1425
292419 79880 9853 1133 206138 47292 5394 653 67260 19054 5545 1624
306300 85520 10984 1233 237521 54829 6645 755 78411 21896 6836 1818
317475 90467 11595 1323 265121 61750 7615 846 89325 24452 7825 2158
331121 94625 12347 1439 282573 67675 8150 962 98670 27462 8382 2368
339483 97047 12904 1545 303488 75017 9102 1065 104301 29991 9364 2560
349159 102867 13595 1588 330309 79544 9497 1104 110864 31497 9772 2732
349159 102867 14006 1675 367116 84747 9911 1189 118743 33121 10191 2955
361546 106528 14495 1727 367116 84747 10398 1237 118743 34954 10687 2955''')
addCounts('rm',80,'''
85472 7990 0 0 0 0 0 0 0 0 0 0
123680 23171 1522 0 6769 710 0 0 604 0 0 0
149660 33068 3119 242 15151 2703 214 16 2348 563 154 39
173329 40096 4333 392 25147 5090 459 49 4158 1142 322 89
197680 47752 5393 528 36011 7406 729 73 6455 1909 566 142
215637 55206 6269 649 54699 11278 1167 104 9348 2729 822 228
232126 59755 7166 762 71609 15228 1620 146 13340 3936 1218 315
245724 65154 8028 874 92251 20449 2199 215 17976 5147 1672 454
266512 69536 8851 969 114899 24444 2766 283 21460 6593 2082 589
284919 75121 9517 1054 168097 30334 3307 367 26644 7960 2499 709
297409 81677 9881 1136 206956 42446 4060 443 37256 10030 3072 850
312776 87682 11053 1187 269329 55320 5318 499 45857 12931 4004 1057
325263 91332 11903 1340 319380 66933 6904 680 54915 15934 4903 1343
332697 95515 12400 1433 354508 78131 8184 811 68632 19198 5779 1656
341258 100760 12987 1501 401990 85495 9365 897 74937 22159 6560 1866
351567 104001 13716 1566 473868 94786 10063 1002 83004 24049 7551 2106
351567 104001 14154 1658 473868 106899 10943 1138 93478 26343 8205 2248
364529 107805 14658 1704 473868 124736 11943 1227 93478 29174 8961 2430
364529 107805 14658 1779 473868 124736 13127 1338 108652 29174 9825 2637
364529 112773 15237 1779 473868 124736 13127 1338 108652 33021 9825 2869''')
addCounts('rm',95,'''
0 0 0 0 0 0 0 0 0 0 0 0
60166 0 0 0 0 0 0 0 0 0 0 0
102198 17824 0 0 4417 0 0 0 0 0 0 0
134179 27359 2427 164 11617 2038 135 0 1669 406 94 15
158503 35208 3630 315 19733 3822 345 31 3302 876 254 71
178746 41438 4642 457 28738 6079 603 68 5291 1471 444 114
197680 48928 5592 573 41331 8602 932 86 7545 2171 707 169
215637 55206 6433 675 58205 11936 1271 123 10468 2949 966 268
232126 59755 7363 790 77308 16260 1831 170 14241 4188 1301 344
245724 65154 8028 903 92251 20449 2368 228 17976 5578 1672 490
253685 69536 8851 969 114899 27022 3033 302 21460 6593 2082 589
277437 72194 9517 1054 160209 30334 3660 367 26644 7960 2499 770
292984 80336 9881 1136 206956 42446 4060 443 35448 10030 3072 850
307233 84508 10841 1187 250312 52221 5318 499 45857 12931 4004 1057
318747 89492 11675 1340 291897 62555 6537 680 54915 15934 4903 1343
325263 93329 12400 1399 354508 72146 7707 764 63329 17985 5779 1565
341258 97975 12987 1501 401990 85495 8758 897 74937 20553 6560 1866
341258 100760 13323 1566 473868 94786 10063 1002 83004 24049 7551 2106
351567 104001 14154 1614 473868 106899 10943 1138 93478 26343 8205 2248
364529 107805 14658 1658 473868 106899 11943 1227 93478 26343 8961 2430''')

# Null counts of zero in our source indicate uncalibrated N in some rows;
# use N mask, not zero, to avoid interpreting missing rules as 0% FPR.
for name in ['ind','rm']:
  for p in [80,95]:
    arr=N[(name,p,'assumed')]
    cnt=counts[(name,p)]
    assert arr.shape==(26,12) and cnt.shape==(26,12)
    assert np.all(cnt[~np.isfinite(arr)]==0),(name,p,'nonzero for missing minN')
    assert np.all((cnt[np.isfinite(arr)]>0))
    assert np.all(arr[np.isfinite(arr)]>=4)
    if p==95:
      ref=N[(name,80,'assumed')]
      same=np.isfinite(arr)&np.isfinite(ref)
      assert np.all(arr[same]>=ref[same])

# z score for a one-sided 95% Wilson interval (reported by the simulation).
def wilson_upper(count,n=800000):
    z=1.644853626951473
    p=count/n
    return (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

threshold_labels={
'ind': ['p < .05','p < .01','p < .001','p < .0001',
        'BF10 >= 1.0688','BF10 >= 4.1857','BF10 >= 30.873','BF10 >= 237.843',
        'BF10 >= 3','BF10 >= 10','BF10 >= 30','BF10 >= 100'],
'rm':['p < .05','p < .01','p < .001','p < .0001',
      'BF10 >= 0.5785','BF10 >= 2.5989','BF10 >= 22.537','BF10 >= 197.638',
      'BF10 >= 3','BF10 >= 10','BF10 >= 30','BF10 >= 100']}

rows=[]
for des in ['ind','rm']:
 for approach in ['assumed','observed']:
  xs=fgrid if approach=='assumed' else bins
  for k,x in enumerate(xs):
   for j,criterion in enumerate(threshold_labels[des]):
    r={'design':des,'calibration':approach,'effect_index':k,
       'effect_value':float(x),'effect_bin':(bin_names[k] if approach=='observed' else ''),'criterion':criterion,
       'criterion_index':j}
    for p in [80,95]:
      r['N'+str(p)]=N[(des,p,approach)][k,j]
      if approach=='assumed':
       q=counts[(des,p)][k,j]/800000
       upper=wilson_upper(counts[(des,p)][k,j])
      else:
       q=obsFPR[(des,p)][j]
       upper=wilson_upper(round(q*800000))
      r['FPR'+str(p)]=q
      r['upper'+str(p)]=upper
      r['pass'+str(p)]=bool(np.isfinite(r['N'+str(p)]) and upper<.05)
    rows.append(r)
results=pd.DataFrame(rows)
results.to_csv(ROOT/'ANOVA_all_persistence_and_FPR_source.csv',index=False)

import matplotlib.ticker as mticker
COLORS=['#137d90','#bb4058','#9d6c17','#6d4ea1']
FAM=[list(range(4)),list(range(8,12)),list(range(4,8))]
FAM_TITLE=['p-value thresholds','Reference Bayes factors','p-matched Bayes factors']


def prettify(label):
 return label.replace('BF10','BF₁₀').replace('>=','≥')

def style_axis(ax):
 ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
 ax.tick_params(labelsize=9.3,length=3.5,direction='out')
 ax.grid(False)

def persistence_plot(des,idx,basename):
 fig,axes=plt.subplots(1,3,figsize=(18.0,6.5),dpi=230)
 x=fgrid if idx=='assumed' else bins
 for k,ax in enumerate(axes):
  for m,j in enumerate(FAM[k]):
   for target,sty,mark in [(80,'-','o'),(95,'--','s')]:
    v=N[(des,target,idx)][:,j]
    ok=np.isfinite(v)&(x>0)
    ax.plot(x[ok],v[ok],color=COLORS[m],linestyle=sty,linewidth=2.05,
      marker=mark,markersize=3.1,markerfacecolor='white' if target==95 else COLORS[m])
  ax.set_xscale('log');ax.set_yscale('log')
  ax.set_xlim((.048 if idx=='observed' else .048),(3.2 if idx=='observed' else 1.06))
  ax.set_ylim(3.7,1150)
  ax.set_xticks(([.05,.1,.2,.5,1,2,3] if idx=='observed' else [.05,.1,.2,.5,1]))
  ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
  ax.set_yticks([4,10,20,50,100,200,500,1000])
  ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
  ax.set_ylabel('Minimum N (per group)' if des=='ind' else 'Minimum N (participants)',fontsize=10.5)
  ax.set_xlabel(('Assumed population effect (Cohen’s f)' if idx=='assumed' else 'Observed f bin (lower boundary)'),fontsize=10.5)
  ax.set_title(f'({chr(65+k)}) {FAM_TITLE[k]}',fontsize=11.5,fontweight='semibold',loc='left')
  handles=[Line2D([0],[0],color=COLORS[i],linewidth=2.5,label=prettify(threshold_labels[des][j])) for i,j in enumerate(FAM[k])]
  ax.legend(handles=handles,fontsize=8.5,frameon=False,loc='upper right')
  style_axis(ax)
 fig.text(.5,.06,'Solid curves: 80% persistence     Dashed curves: 95% persistence',ha='center',fontsize=12)
 title=('Independent-groups ANOVA' if des=='ind' else 'Repeated-measures ANOVA')
 fig.suptitle(f'{title} — minimum sample size by {"assumed" if idx=="assumed" else "observed"} effect size',fontsize=17,weight='bold',y=.965)
 fig.subplots_adjust(left=.06,right=.985,top=.83,bottom=.19,wspace=.22)
 fig.savefig(ROOT/(basename+'.png'),dpi=235,facecolor='white')
 fig.savefig(ROOT/(basename+'.svg'),facecolor='white')
 plt.close(fig)


def combined_observed():
 fig,axes=plt.subplots(2,3,figsize=(18.0,10.1),dpi=230)
 for rr,des in enumerate(['ind','rm']):
  for k in range(3):
   ax=axes[rr,k]
   for m,j in enumerate(FAM[k]):
    for target,sty,mark in [(80,'-','o'),(95,'--','s')]:
     v=N[(des,target,'observed')][:,j]
     ok=np.isfinite(v)&(bins>0)
     ax.plot(bins[ok],v[ok],color=COLORS[m],linestyle=sty,linewidth=2.0,
      marker=mark,markersize=3,markerfacecolor='white' if target==95 else COLORS[m])
   ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.048,3.2);ax.set_ylim(3.7,1150)
   ax.set_xticks([.05,.1,.2,.5,1,2,3]);ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
   ax.set_yticks([4,10,20,50,100,200,500,1000]);ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
   ax.set_xlabel('Observed Cohen’s f (bin lower boundary)',fontsize=10.5)
   ax.set_ylabel('Participants per group' if des=='ind' else 'Paired/repeated participants',fontsize=10.5)
   ax.set_title(f'({chr(65+rr*3+k)}) {"Independent" if rr==0 else "Repeated measures"}: {FAM_TITLE[k]}',fontsize=11,weight='semibold',loc='left')
   handles=[Line2D([0],[0],color=COLORS[i],linewidth=2,label=prettify(threshold_labels[des][j])) for i,j in enumerate(FAM[k])]
   ax.legend(handles=handles,fontsize=7.9,frameon=False,loc='upper right')
   style_axis(ax)
 fig.suptitle('Observed-effect calibrations — minimum sample size for persistence',fontsize=18,weight='bold',y=.984)
 fig.text(.5,.024,'Solid curves: 80% persistence     Dashed curves: 95% persistence',ha='center',fontsize=12)
 fig.subplots_adjust(left=.062,right=.992,top=.921,bottom=.09,hspace=.37,wspace=.24)
 for ext in ['png','svg']:
  fig.savefig(ROOT/('ANOVA_Figure_3_observed_persistence_all12.'+ext),dpi=240,facecolor='white')
 plt.close(fig)


def assumed_fpr():
 fig,axes=plt.subplots(2,3,figsize=(18.,10.1),dpi=240)
 for rr,des in enumerate(['ind','rm']):
  for k in range(3):
   ax=axes[rr,k]
   for m,j in enumerate(FAM[k]):
    for target,sty,mark in [(80,'-','o'),(95,'--','s')]:
     y=counts[(des,target)][:,j]/800000*100
     mask=np.isfinite(N[(des,target,'assumed')][:,j])&(y>0)
     ax.plot(fgrid[mask],y[mask],color=COLORS[m],linestyle=sty,linewidth=2.0,
        marker=mark,markersize=3.0,markerfacecolor='white' if target==95 else COLORS[m])
   ax.axhline(5,color='#555555',linestyle=':',linewidth=1.3)
   ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.05,1.05);ax.set_ylim(.0008,80)
   ax.set_xticks([.05,.1,.2,.5,1]);ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
   ax.set_yticks([.001,.01,.1,1,5,10,50]);ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))
   ax.set_xlabel('Assumed population effect (Cohen’s f)',fontsize=10.5)
   ax.set_ylabel('Overall false-positive rate (%)',fontsize=10.5)
   ax.set_title(f'({chr(65+rr*3+k)}) {"Independent" if rr==0 else "Repeated measures"}: {FAM_TITLE[k]}',fontsize=11,weight='semibold',loc='left')
   handles=[Line2D([0],[0],color=COLORS[i],linewidth=2,label=prettify(threshold_labels[des][j])) for i,j in enumerate(FAM[k])]
   ax.legend(handles=handles,fontsize=7.9,frameon=False,loc='upper left')
   style_axis(ax)
 fig.suptitle('Sequential false-positive rates — assumed-effect stopping rules',fontsize=18,weight='bold',y=.985)
 fig.text(.5,.025,'Solid curves: 80% persistence     Dashed curves: 95% persistence     Dotted horizontal line: 5% FPR',ha='center',fontsize=11)
 fig.subplots_adjust(left=.064,right=.993,top=.922,bottom=.089,hspace=.37,wspace=.24)
 for ext in ['png','svg']:
  fig.savefig(ROOT/('ANOVA_Figure_4_assumed_effect_FPR_all12.'+ext),dpi=240,facecolor='white')
 plt.close(fig)


def observed_validation():
 # There is no observed-effect bin-level NULL simulation output for ANOVA.
 # Display genuine total error by THRESHOLD, not fabricated per-bin rates.
 fig,axes=plt.subplots(2,3,figsize=(18.,10.1),dpi=230)
 for rr,des in enumerate(['ind','rm']):
  for k in range(3):
   ax=axes[rr,k];inds=FAM[k]
   X=np.arange(4)
   for target,ls,mark,color in [(80,'-','o','#0a7892'),(95,'--','s','#c24d35')]:
    y=obsFPR[(des,target)][inds]*100
    ax.plot(X,y,color=color,lw=2.4,ls=ls,marker=mark,markersize=7,markerfacecolor='white',label=f'{target}% persistence')
   # naive rates displayed as unconnected small gray circles
   ax.scatter(X,naive[des][inds]*100,marker='x',color='#565656',s=50,label='Unrestricted first crossing')
   ax.axhline(5,ls=':',color='#555555',lw=1.35)
   ax.set_yscale('log');ax.set_ylim(.008,120)
   ax.set_xlim(-.35,3.35);ax.set_xticks(X)
   ax.set_xticklabels([prettify(threshold_labels[des][j]).replace('BF₁₀ >= ','BF≥') for j in inds],rotation=18,ha='right',fontsize=9)
   ax.set_yticks([.01,.1,1,5,10,50]);ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))
   ax.set_ylabel('Overall false-positive rate (%)',fontsize=10.5)
   ax.set_title(f'({chr(65+rr*3+k)}) {"Independent" if rr==0 else "Repeated measures"}: {FAM_TITLE[k]}',fontsize=11,weight='semibold',loc='left')
   if rr==0 and k==0:ax.legend(loc='upper right',fontsize=8.5,frameon=False)
   style_axis(ax)
 fig.suptitle('Observed-effect adaptive rules — total sequential false-positive rates',fontsize=18,weight='bold',y=.984)
 fig.text(.5,.02,'Each point is the TOTAL null stopping rate across every observed-effect bin; 5% line is a visual reference.',ha='center',fontsize=10.5)
 fig.subplots_adjust(left=.063,right=.995,top=.92,bottom=.12,hspace=.52,wspace=.23)
 for ext in ['png','svg']:
  fig.savefig(ROOT/('ANOVA_Figure_5_observed_rule_overall_FPR_all12.'+ext),dpi=240,facecolor='white')
 plt.close(fig)

persistence_plot('ind','assumed','ANOVA_Figure_1_independent_assumed_persistence')
persistence_plot('rm','assumed','ANOVA_Figure_2_repeated_assumed_persistence')
combined_observed()
assumed_fpr()
observed_validation()
print('Generated 5 verified ANOVA figures and analysis source CSV.')
print('Sample validation:')
for des in ['ind','rm']:
 for idx in [16,20,23,25]:
  print(des,round(fgrid[idx],3),'p001 P95 N',N[(des,95,'assumed')][idx,2],
    'BF>=10 P95 N',N[(des,95,'assumed')][idx,9],
    'p001 FPR95',round(counts[(des,95)][idx,2]/8000,3),
    'BF>=10 FPR95',round(counts[(des,95)][idx,9]/8000,3))
