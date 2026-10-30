# Run targeted balanced variance heterogeneity and sphericity checks across all 12 evidence thresholds.
import numpy as np,pandas as pd,re,os
ROOT=os.path.dirname(os.path.abspath(__file__))
from scipy.optimize import brentq
source=open(os.path.join(ROOT,'focused_all_thresholds_simulation.py'),encoding='utf-8').read()
namespace={}
exec(compile(source.split('output_file=ROOT+')[0],'<shared>', 'exec'),namespace)

def noise(rng,b):
 return rng.standard_normal((b,1000))

def altered_data(rng,dist,code,n,b,g,k,repeated):
 ix=namespace['N_GRID']-1
 if dist=='variance9':
  sums=[];ss=np.zeros((b,len(ix)));es=np.zeros_like(ss)
  for var in [1.,1.,9.]:
   xx=noise(rng,b)*np.sqrt(var)
   s=np.cumsum(xx,axis=1)[:,ix];q=np.cumsum(xx**2,axis=1)[:,ix]
   sums.append(s);es+=q-s*s/n;ss+=s
  bg=(sum(s*s for s in sums)-ss*ss/g)/n
  mse=np.maximum(es/(g*(n-1)),1e-12)
 else:
  target=float(dist.replace('sphericity',''))/100
  def epsilon(r):return (1+r)**2/(2*(1+r*r))
  rr=brentq(lambda r:epsilon(r)-target,1+1e-9,1e6)
  lam=2*np.array([1,rr])/(1+rr)
  bg=np.zeros((b,len(ix)));er=np.zeros_like(bg)
  for v in lam:
   xx=noise(rng,b)*np.sqrt(v)
   s=np.cumsum(xx,axis=1)[:,ix];q=np.cumsum(xx*xx,axis=1)[:,ix]
   bg+=s*s/n;er+=q-s*s/n
  mse=np.maximum(er/(k*(n-1)),1e-12)
 f=(bg/k)/mse
 eff=np.sqrt(bg/(g*n*mse))
 return f,eff
namespace['f_data']=altered_data
out=[];N=24000
for design,code,g,k,repeated,dist in [
 ('One-way independent ANOVA','one_ind',3,2,False,'variance9'),
 ('One-way repeated ANOVA','one_rep',3,2,True,'sphericity55')]:
 mn,edges=namespace['get_minimum_N'](design,code)
 CUT,match=namespace['build_cutoffs'](g,k,repeated)
 n=namespace['simulate'](design,code,g,k,repeated,dist,CUT,mn,edges,N)
 print(design,dist,'p05',n[0]/N*100,'p001',n[2]/N*100,'bf3',n[8]/N*100,'bf10',n[9]/N*100,flush=True)
 for i,c in enumerate(n):
  out.append(dict(design=design,code=code,distribution=dist,criterion_index=i,
   criterion=namespace['CUT_NAMES'][i],family=namespace['FAMILY'][i],
   threshold_value=np.r_[namespace['P'],match,namespace['BFS']][i],
   n_null=N,n_false_stops=int(c),FPR_pct=100*c/N,
   Wilson_one_sided95_upper_pct=100*namespace['wilson_upper'](c,N),
   passes_5pct=namespace['wilson_upper'](c,N)<.05,
   status='exploratory; original 95% persistence minimum N; not refitted under violation'))
pd.DataFrame(out).to_csv(os.path.join(ROOT,'focused_extra_violations_all_thresholds.csv'),index=False)
