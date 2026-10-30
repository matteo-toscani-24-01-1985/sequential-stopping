"""Focused exploratory robustness check of original observed-effect persistence stopping rules.
Replicates sequential H0 testing under normal / Student t3 / standardized lognormal errors.
Original minimum-N grids are read from manuscript-derived CSV data. Independent groups balanced.
Uses p and Zellner-Siow/JZS BF critical F values at every preplanned checkpoint.
Run: python focused_all_thresholds_simulation.py [num_null] (default 20000)
"""
import os,sys,re,time
import numpy as np
import pandas as pd
from scipy.special import roots_legendre
from scipy.stats import f as Fdist

ROOT=os.path.dirname(os.path.abspath(__file__))
NREP=int(sys.argv[1]) if len(sys.argv)>1 else 24000
BATCH=240
N_GRID=np.r_[np.arange(4,41,2),np.arange(50,201,10),np.arange(220,1001,20)]
P=[.05,.01,.001,.0001]
BFS=[3,10,30,100]
CUT_NAMES=['p < .05','p < .01','p < .001','p < .0001', 'BF n50 ~ p=.05','BF n50 ~ p=.01','BF n50 ~ p=.001','BF n50 ~ p=.0001', 'BF >= 3','BF >= 10','BF >= 30','BF >= 100']
FAMILY=['p']*4+['BF-matched']*4+['BF-reference']*4
SPEC=[
 ('Independent t-test','t_ind',2,1,False,'Figure_3_source_independent_95pct.csv'),
 ('Paired t-test','t_pair',2,1,True,'Figure_3_source_paired_95pct.csv'),
 ('One-way independent ANOVA','one_ind',3,2,False,None),
 ('One-way repeated ANOVA','one_rep',3,2,True,None),
 ('2x2 independent interaction','fact2_ind',4,1,False,None),
 ('2x2 repeated interaction','fact2_rep',4,1,True,None),
 ('2x2x2 independent interaction','fact3_ind',8,1,False,None),
 ('2x2x2 repeated interaction','fact3_rep',8,1,True,None),
]
DISTS=['normal','t3','lognormal']
# 160-node Gaussian quadrature of the half-normal prior on [0,12], negligible tail.
_nodes,_wg=roots_legendre(160)
Z=6*(_nodes+1)
W=6*_wg*np.sqrt(2/np.pi)*np.exp(-.5*Z**2)

def bf_of_f(f,n,g,k,repeated):
 n=np.asarray(n,dtype=float).reshape(-1)
 f=np.asarray(f,dtype=float).reshape(-1)
 df2=(n-1)*(k if repeated else g)
 neff=n if repeated else n/g
 h=Z[:,None]**2/(Z[:,None]**2+neff[None,:]*.5)
 q=k*f[None,:]/df2[None,:]
 logt=.5*k*np.log(h)+.5*(k+df2[None,:])*(np.log1p(q)-np.log1p(h*q))
 return (W[:,None]*np.exp(logt)).sum(axis=0)

def build_cutoffs(g,k,rep):
 n=N_GRID.astype(float)
 df2=(n-1)*(k if rep else g)
 pv=Fdist.isf(np.asarray(P)[:,None],k,df2[None,:])
 f50=Fdist.isf(np.asarray(P),k,np.repeat((50-1)*(k if rep else g),4))
 bf50=bf_of_f(f50,np.repeat(50.,4),g,k,rep)
 allbf=np.r_[bf50,BFS]
 cut=[v for v in pv]
 for target in allbf:
  lo=np.zeros(len(n));hi=np.repeat(20.,len(n))
  for j in range(12):
   mask=bf_of_f(hi,n,g,k,rep)<target
   if not mask.any():break
   hi[mask]*=2
  for j in range(40):
   md=(lo+hi)/2
   mask=bf_of_f(md,n,g,k,rep)<target
   lo[mask]=md[mask];hi[~mask]=md[~mask]
  cut.append((lo+hi)/2)
 return np.stack(cut,axis=0), bf50

def norm_bin(s):
 m=re.search(r'\[\s*([0-9.]+)\s*,\s*([0-9.]+|Inf)\s*\)',str(s))
 return (float(m[1]),np.inf if m[2]=='Inf' else float(m[2])) if m else None

def get_minimum_N(design,code):
 if code.startswith('t_'):
  file='Figure_3_source_independent_95pct.csv' if code=='t_ind' else 'Figure_3_source_paired_95pct.csv'
  x=pd.read_csv(os.path.join(ROOT,file)); cols=list(x.columns);N=x.iloc[:,1:13].to_numpy(dtype=float)
  edges=np.array([norm_bin(t)[0] for t in x.iloc[:,0]]+[norm_bin(x.iloc[-1,0])[1]])
  return N,edges
 if code.startswith('one_'):
  allx=pd.read_csv(ROOT+'/ANOVA_all_persistence_and_FPR_source.csv')
  code2='ind' if code=='one_ind' else 'rm'
  x=allx[(allx.design==code2)&(allx.calibration=='observed')]
  labels=sorted(x.effect_bin.dropna().unique(),key=lambda t:norm_bin(t)[0]);N=np.full((len(labels),12),np.nan)
  for bi,label in enumerate(labels):
   ss=x[x.effect_bin==label]
   for _,r in ss.iterrows():N[bi,int(r.criterion_index)]=r.N95 if bool(r.pass95) else np.nan
  edges=np.array([norm_bin(t)[0] for t in labels]+[norm_bin(labels[-1])[1]])
  return N,edges
 from factorial_full_observed_minN_data import MINN,EDGES
 return MINN[code].copy(),EDGES.copy()

def rng_noise(rng,dist,shape):
 if dist=='normal':return rng.standard_normal(shape)
 if dist=='t3':return rng.standard_t(3,size=shape)/np.sqrt(3.)
 # Standardized lognormal with fixed expected mean and variance
 z=rng.standard_normal(shape)
 return (np.exp(z)-np.exp(.5))/np.sqrt((np.exp(1)-1)*np.exp(1))

def f_data(rng,dist,code,n,batch,g,k,repeated):
 ix=N_GRID-1
 if code=='t_ind':
  X=rng_noise(rng,dist,(batch,1000));Y=rng_noise(rng,dist,(batch,1000))
  xs=np.cumsum(X,axis=1)[:,ix]; ys=np.cumsum(Y,axis=1)[:,ix]
  xx=np.cumsum(X*X,axis=1)[:,ix];yy=np.cumsum(Y*Y,axis=1)[:,ix]
  df2=2*(n-1)
  MSE=np.maximum((xx+yy-(xs*xs+ys*ys)/n)/df2,1e-12)
  F=((ys-xs)/n)**2/(MSE*2/n)
  effect=np.sqrt(F*2/n)
  return F,effect
 if code=='t_pair':
  X=rng_noise(rng,dist,(batch,1000))
  xs=np.cumsum(X,axis=1)[:,ix];xx=np.cumsum(X*X,axis=1)[:,ix]
  MSE=np.maximum((xx-xs*xs/n)/(n-1),1e-12)
  F=(xs/n)**2/(MSE/n)
  effect=np.sqrt(F/n)
  return F,effect
 if repeated:
  # independent repeated-condition contrasts, as in robustness engine
  target=np.zeros((batch,n.shape[1]));er=np.zeros_like(target)
  for _ in range(k):
   X=rng_noise(rng,dist,(batch,1000))
   s=np.cumsum(X,axis=1)[:,ix];q=np.cumsum(X*X,axis=1)[:,ix]
   target+=s*s/n;er+=q-s*s/n
  MSE=np.maximum(er/(k*(n-1)),1e-12)
  F=(target/k)/MSE
  effect=np.sqrt(target/(g*n*MSE))
  return F,effect
 # independent balanced groups
 er=np.zeros((batch,n.shape[1]));ss=np.zeros((batch,n.shape[1]))
 sums=[]
 for _ in range(g):
  X=rng_noise(rng,dist,(batch,1000))
  s=np.cumsum(X,axis=1)[:,ix];q=np.cumsum(X*X,axis=1)[:,ix]
  er+=q-s*s/n; sums.append(s)
  ss+=s
 MSE=np.maximum(er/(g*(n-1)),1e-12)
 if k==g-1:
  bg=(sum(s*s for s in sums)-ss*ss/g)/n
 else:
  # 2x2 or 2x2x2 highest-order, normalized single contrast
  proj=sum(s * ((-1)**(j.bit_count())/np.sqrt(g)) for j,s in enumerate(sums))
  bg=proj*proj/n
 F=(bg/k)/MSE
 effect=np.sqrt(bg/(g*n*MSE))
 return F,effect

def simulate(design,code,g,k,repeated,dist,CUT,minN,edges,NREP):
 rng=np.random.default_rng(20261029+sum(map(ord,code))*13+sum(map(ord,dist))*7)
 total=np.zeros(12,dtype=np.int64)
 n=N_GRID.astype(float)[None,:]
 for start in range(0,NREP,BATCH):
  b=min(BATCH,NREP-start)
  F,effect=f_data(rng,dist,code,n,b,g,k,repeated)
  bi=np.searchsorted(edges,effect,side='right')-1
  valid=(bi>=0)&(bi<minN.shape[0])
  bi=bi.clip(0,minN.shape[0]-1)
  # size (b,looks,threshold)
  mN=np.take(minN,bi,axis=0)
  eligible=np.isfinite(mN)&(n[:,:,None]>=mN)
  # F threshold for each test, at each checkpoint, shape thresholds x checkpoints
  passed=F[:,:,None]>=CUT.T[None,:,:]
  stops=(eligible&passed).any(axis=1)
  total+=stops.sum(axis=0)
 return total

def wilson_upper(k,n,z=1.644853626951473):
 p=k/n;return (p+z*z/(2*n)+z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

output_file=ROOT+'/focused_all_thresholds_robustness_results.csv'
records=pd.read_csv(output_file).to_dict('records') if os.path.exists(output_file) else []
start=time.time()
for design,code,g,k,rep,_ in SPEC:
 minN,edges=get_minimum_N(design,code)
 CUT,match=build_cutoffs(g,k,rep)
 print('Start',design,'minN shape',minN.shape,'BF n50',np.round(match,4),flush=True)
 for dist in DISTS:
  if not code.startswith('fact') and any(r['code']==code and r['distribution']==dist for r in records):
   print('  skip complete',dist,flush=True);continue
  records=[r for r in records if not (r['code']==code and r['distribution']==dist)]
  counts=simulate(design,code,g,k,rep,dist,CUT,minN,edges,NREP)
  for j,c in enumerate(counts):
   records.append({'design':design,'code':code,'distribution':dist,'criterion_index':j,
    'criterion':CUT_NAMES[j], 'family':FAMILY[j], 'threshold_value':np.r_[P,match,BFS][j],
    'n_null':NREP,'n_false_stops':int(c),'FPR_pct':100*c/NREP,
    'Wilson_one_sided95_upper_pct':100*wilson_upper(c,NREP),
    'passes_5pct':bool(wilson_upper(c,NREP)<.05),
    'status':'exploratory; original 95% persistence minimum N; not refitted under violation'})
  print(' ',dist, 'p05',round(100*counts[0]/NREP,2),'bf3',round(100*counts[8]/NREP,2),'p001',round(100*counts[2]/NREP,2),'bf10',round(100*counts[9]/NREP,2),'elapsed',round(time.time()-start,1),flush=True)
 pd.DataFrame(records).to_csv(ROOT+'/focused_all_thresholds_robustness_results.csv',index=False)
print('Finished',len(records),'rows',round(time.time()-start,1),'sec')
