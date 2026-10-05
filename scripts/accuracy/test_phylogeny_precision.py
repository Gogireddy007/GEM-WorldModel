#!/usr/bin/env python
"""This checks whether it matters that labeled species get species-level phylogeny
vectors while most GEM genomes can only be placed at genus or family level, by
training and testing at different precision levels with species held out.
"""

import numpy as np, pandas as pd, re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from gem_worldmodel.features import taxon_vectors as tv, rrna16s
from gem_worldmodel.utils.config import load_config, resolve_path
dc=load_config('data'); fc=load_config('features'); proc=resolve_path(dc['paths']['processed_dir']); k=fc['rrna16s']['kmer_k']
df=pd.read_csv(proc/'features_sample_expanded.csv'); seqs=pd.read_csv(proc/'labeled_16s_sequences.csv').fillna('')
prof={s:rrna16s.kmer_profile(q,k) for s,q in zip(seqs.species,seqs.sequence) if q}
landmarks=list(np.load(proc/'labeled_16s_landmark_names.npy'))
S=np.vstack([np.array([rrna16s.kmer_cosine_distance(prof[s],prof[l]) for l in landmarks]) if s in prof else np.full(16,np.nan) for s in df.species])
S=np.where(np.isnan(S),np.nanmedian(S,axis=0),S)
table=tv.load_taxon_table(proc/'gtdb_taxon_vectors.csv')
def LMat(maxrank, only=None):
    out=[]
    for t in df.gtdb_taxonomy:
        if only=='genus':   # force genus-level (skip species) like a GEM genome with an unmatched species name
            t2=re.sub(r's__[^;]*','s__',t)
        elif only=='family':
            t2=re.sub(r's__[^;]*','s__',re.sub(r'g__[^;]*','g__',t))
        else: t2=t
        out.append(tv.lookup(t2,table,maxrank)[0])
    return np.vstack(out)
LM_sp=LMat('family'); LM_g=LMat('family','genus'); LM_f=LMat('family','family')
T=df[['genome_size_bp','gc_content','trna_count']].to_numpy(float); y=df['doubling_time_hours_ref'].to_numpy(); tl=np.log(y)
def X(LM): return np.hstack([T,LM,S])
def run(train_LMs, test_LM, seeds=(42,1,7,13,2024)):
    out=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); ug=np.unique(df.species); perm=dict(zip(ug,rng.permutation(len(ug)))); g=np.array([perm[v] for v in df.species]); oof=np.full(len(df),np.nan)
        for tr,te in GroupKFold(5).split(T,tl,g):
            Xtr=np.vstack([X(L)[tr] for L in train_LMs]); ytr=np.concatenate([tl[tr]]*len(train_LMs))
            oof[te]=GradientBoostingRegressor(random_state=seed,n_estimators=200,max_depth=3).fit(Xtr,ytr).predict(X(test_LM)[te])
        out.append((r2_score(tl,oof),spearmanr(tl,oof).correlation,np.exp(np.median(np.abs(tl-oof)))))
    return np.array(out).mean(0)
print('SPECIES held out. Phylogeny precision of training rows vs test rows        R2(log) Spearman fold-err')
for name,tr,te in [('train species-level, test species-level (what we had)',[LM_sp],LM_sp),
                   ('train species-level, test GENUS-level (deployment mismatch)',[LM_sp],LM_g),
                   ('train GENUS-level,   test GENUS-level',[LM_g],LM_g),
                   ('train mixed (species+genus+family), test GENUS-level',[LM_sp,LM_g,LM_f],LM_g),
                   ('train species-level, test FAMILY-level (deployment mismatch)',[LM_sp],LM_f),
                   ('train mixed (species+genus+family), test FAMILY-level',[LM_sp,LM_g,LM_f],LM_f)]:
    r=run(tr,te); print('  %-62s %6.3f %8.3f %7.2fx' % (name,*r))
