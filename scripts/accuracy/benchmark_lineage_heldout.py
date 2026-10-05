#!/usr/bin/env python
"""This tests the deployment feature set (genome traits, GTDB landmark phylogeny,
16S landmark distances) when whole species, genera, families or orders are held
out, which is the situation for most GEM genomes. Needs
labeled_16s_sequences.csv (fetch_labeled_16s.py) and gtdb_taxon_vectors.csv
(build_gtdb_landmark_features.py). Also writes the 16S landmark names used by
the prediction script.
"""

import numpy as np, pandas as pd, re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from gem_worldmodel.features import taxon_vectors as tv, rrna16s
from gem_worldmodel.utils.config import load_config, resolve_path
dc=load_config('data'); fc=load_config('features'); proc=resolve_path(dc['paths']['processed_dir'])
k=fc['rrna16s']['kmer_k']
df=pd.read_csv(proc/'features_sample_expanded.csv'); seqs=pd.read_csv(proc/'labeled_16s_sequences.csv').fillna('')
seq=dict(zip(seqs.species,seqs.sequence)); prof={s:rrna16s.kmer_profile(q,k) for s,q in seq.items() if q}
names=list(prof); mat,_=rrna16s.build_16s_distance_matrix_from_profiles(prof)
# farthest-point landmarks among labeled 16S profiles
rng=np.random.default_rng(0); cur=int(rng.integers(len(names))); chosen=[]; mind=np.full(len(names),np.inf)
for _ in range(16):
    chosen.append(cur); mind=np.minimum(mind,mat[cur]); cur=int(np.argmax(mind))
landmarks=[names[i] for i in chosen]; print('16S landmarks:',landmarks[:4],'...')
def vec16(profile):
    return np.array([rrna16s.kmer_cosine_distance(profile,prof[l]) for l in landmarks])
S=np.vstack([vec16(prof[s]) if s in prof else np.full(16,np.nan) for s in df.species])
print('rows with 16S vector: %d of %d' % ((~np.isnan(S).any(axis=1)).sum(), len(df)))
S=np.where(np.isnan(S),np.nanmedian(S,axis=0),S)
table=tv.load_taxon_table(proc/'gtdb_taxon_vectors.csv')
LM=np.vstack([tv.lookup(t,table,'family')[0] for t in df['gtdb_taxonomy']])
T=df[['genome_size_bp','gc_content','trna_count']].to_numpy(float)
y=df['doubling_time_hours_ref'].to_numpy(); tl=np.log(y)
tok=lambda t,p:(re.search(p+r'[^;]*',str(t)).group(0) if re.search(p+r'[^;]*',str(t)) else 'none')
sets={'traits + landmark phylogeny':np.hstack([T,LM]),'traits + 16S landmarks':np.hstack([T,S]),'traits + landmark phylogeny + 16S landmarks':np.hstack([T,LM,S])}
def run(X,groups,seeds=(42,1,7,13,2024)):
    out=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); ug=np.unique(groups); perm=dict(zip(ug,rng.permutation(len(ug)))); g=np.array([perm[v] for v in groups]); oof=np.full(len(df),np.nan)
        for tr,te in GroupKFold(5).split(X,tl,g): oof[te]=GradientBoostingRegressor(random_state=seed,n_estimators=200,max_depth=3).fit(X[tr],tl[tr]).predict(X[te])
        out.append((r2_score(tl,oof),spearmanr(tl,oof).correlation,np.exp(np.median(np.abs(tl-oof)))))
    return np.array(out).mean(0)
for label,col in [('SPECIES held out',df.species.to_numpy()),('GENUS held out',df.gtdb_taxonomy.map(lambda t:tok(t,'g__')).to_numpy()),('FAMILY held out',df.gtdb_taxonomy.map(lambda t:tok(t,'f__')).to_numpy()),('ORDER held out',df.gtdb_taxonomy.map(lambda t:tok(t,'o__')).to_numpy())]:
    print('\n%-18s                                   R2(log) Spearman fold-err' % label)
    for n,X in sets.items():
        r=run(X,col); print('  %-44s %7.3f %8.3f %7.2fx' % (n,*r))
np.save(proc/'labeled_16s_landmark_names.npy',np.array(landmarks))
