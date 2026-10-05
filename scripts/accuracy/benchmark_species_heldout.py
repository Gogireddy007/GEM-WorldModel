#!/usr/bin/env python
"""This compares our model with the Phydon and gRodon reimplementations when whole
species are held out of training. 88 species in the labeled corpus appear as
more than one genome with an identical growth rate, so a random split lets a
species' twin sit in training while it is tested; holding species out removes
that leak. Reports R2 in hours and in log space, Spearman, and the median
fold error.
"""

import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from gem_worldmodel.features import taxon_vectors as tv
from gem_worldmodel.training.baselines import GRodonBaseline, PhydonBaseline
from gem_worldmodel.training.dataset import build_branch_tensors
from gem_worldmodel.training.raw_baseline import concat_branch_tensors
from gem_worldmodel.utils.config import load_config, resolve_path
dc, mc = load_config('data'), load_config('model'); proc=resolve_path(dc['paths']['processed_dir'])
df = pd.read_csv(proc/'features_sample_expanded.csv')
y=df['doubling_time_hours_ref'].to_numpy(); tl=np.log(y); sp=df['species'].to_numpy()
table = tv.load_taxon_table(proc/'gtdb_taxon_vectors.csv')
LM=np.vstack([tv.lookup(t,table,'family')[0] for t in df['gtdb_taxonomy']])
T=df[['genome_size_bp','gc_content','trna_count']].to_numpy(float)
t,_=build_branch_tensors(df,mc,branches=['genomic_traits','gtdb_distance','rrna16s']); allx=concat_branch_tensors(t).numpy()
X1=np.hstack([T,LM]); X2=allx
def folds(seed):
    rng=np.random.default_rng(seed); ug=np.unique(sp); perm=dict(zip(ug,rng.permutation(len(ug))))
    fid=np.full(len(df),-1)
    for k,(tr,te) in enumerate(GroupKFold(5).split(df,tl,np.array([perm[v] for v in sp]))): fid[te]=k
    return fid
rows=[]
for seed in (42,1,7,13,2024):
    fid=folds(seed); P={}
    for name,X in (('ours: traits + landmark phylogeny',X1),('ours: all three branches (MDS)',X2)):
        o=np.full(len(df),np.nan)
        for k in range(5):
            m=fid==k; o[m]=GradientBoostingRegressor(random_state=seed,n_estimators=200,max_depth=3).fit(X[~m],tl[~m]).predict(X[m])
        P[name]=np.exp(o)
    for name,cls in (('Phydon (reimplementation)',PhydonBaseline),('gRodon (reimplementation)',GRodonBaseline)):
        o=np.full(len(df),np.nan)
        for k in range(5):
            m=fid==k; o[m]=cls().fit(df[~m]).predict(df[m])
        P[name]=o
    for name,p in P.items():
        ok=~np.isnan(p)&(p>0)
        rows.append((name,seed,ok.sum(),r2_score(y[ok],p[ok]),r2_score(np.log(y[ok]),np.log(p[ok])),spearmanr(y[ok],p[ok]).correlation,np.exp(np.median(np.abs(np.log(y[ok])-np.log(p[ok]))))))
r=pd.DataFrame(rows,columns=['model','seed','n','r2_hours','r2_log','spearman','fold_err'])
print('SPECIES HELD OUT (no species in both training and test), 5 seeds, mean')
print(r.groupby('model')[['n','r2_hours','r2_log','spearman','fold_err']].mean().round(3).to_string())
