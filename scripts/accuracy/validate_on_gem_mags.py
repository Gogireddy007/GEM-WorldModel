#!/usr/bin/env python
"""This checks the deployment pipeline on real MAGs: GEM genomes whose GTDB species
matches a labeled species are predicted with that species held out of training,
and compared with the prediction for the same species made from its isolate
genome. Small by necessity, only a few labeled species have a GEM genome.
"""

import numpy as np, pandas as pd, re
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from gem_worldmodel.features import taxon_vectors as tv
from gem_worldmodel.utils.config import load_config, resolve_path
dc = load_config('data'); proc = resolve_path(dc['paths']['processed_dir'])
lab = pd.read_csv(proc/'features_sample_expanded.csv')
table = tv.load_taxon_table(proc/'gtdb_taxon_vectors.csv')
tok = lambda t,p: (re.search(p+r'[^;]*',str(t)).group(0) if re.search(p+r'[^;]*',str(t)) else None)
lab['sp']=lab['gtdb_taxonomy'].map(lambda t: tok(t,'s__'))
# GEM genomes
meta = pd.read_csv('data/raw/gem_genome_metadata.tsv', sep='\t', low_memory=False)
gem = pd.read_csv(proc/'unlabeled_corpus_features.csv', low_memory=False)[['genome_id','genome_size_bp','trna_count']]
enr = pd.read_csv(proc/'unlabeled_corpus_features_enriched.csv')[['genome_id','gc_content']]
gem = gem.merge(enr,on='genome_id').merge(meta[['genome_id','ecosystem','completeness','contamination','mimag_quality']],on='genome_id')
gem['sp']=gem['ecosystem'].map(lambda t: tok(t,'s__'))
gem = gem[gem.sp.isin(set(lab.sp.dropna()))].dropna(subset=['genome_size_bp','trna_count','gc_content'])
print('GEM genomes whose GTDB species matches a labeled species: %d genomes, %d species' % (len(gem), gem.sp.nunique()))
print(gem.mimag_quality.value_counts().to_dict())
gem = gem[gem.completeness>=90]; gem=gem[gem.contamination<=5]
print('after completeness>=90 and contamination<=5: %d genomes, %d species' % (len(gem), gem.sp.nunique()))
def build(df, taxcol):
    T = df[['genome_size_bp','gc_content','trna_count']].to_numpy(float)
    LM = np.vstack([ (tv.lookup(t, table,'family')[0] if tv.lookup(t, table,'family')[0] is not None else np.full(16,np.nan)) for t in df[taxcol]])
    return np.hstack([T,LM])
Xl = build(lab,'gtdb_taxonomy'); yl=np.log(lab['doubling_time_hours_ref'].to_numpy())
gem['tax']=gem['ecosystem']; Xg = build(gem,'tax')
okg=~np.isnan(Xg).any(axis=1); gem=gem[okg].reset_index(drop=True); Xg=Xg[okg]
print('MAGs with usable features:', len(gem))
species=lab['sp'].fillna(lab['species']).to_numpy()
res=[]
for seed in [42,1,7]:
    rng=np.random.default_rng(seed); ug=np.unique(species); perm=dict(zip(ug,rng.permutation(len(ug)))); g=np.array([perm[v] for v in species])
    iso=np.full(len(lab),np.nan); mag_pred=[]
    for tr,te in GroupKFold(5).split(Xl,yl,g):
        m=GradientBoostingRegressor(random_state=seed,n_estimators=200,max_depth=3,loss="quantile",alpha=0.5).fit(Xl[tr],yl[tr])
        iso[te]=m.predict(Xl[te])
        te_sp=set(species[te]); mask=gem.sp.isin(te_sp).to_numpy()
        for i in np.where(mask)[0]: mag_pred.append((i,m.predict(Xg[i:i+1])[0]))
    mp=pd.Series(dict(mag_pred)); gm=gem.loc[mp.index].copy(); gm['pred']=mp.values
    truth=lab.drop_duplicates('sp').set_index('sp')['doubling_time_hours_ref'].map(np.log)
    gm['truth']=gm.sp.map(truth)
    # isolate predictions for the same species
    lab_iso=pd.DataFrame({'sp':lab.sp,'iso':iso,'truth':yl}).dropna(subset=['sp']).drop_duplicates('sp').set_index('sp')
    sp_mag=gm.groupby('sp')['pred'].median()
    common=sp_mag.index.intersection(lab_iso.index)
    t=lab_iso.loc[common,'truth']; a=lab_iso.loc[common,'iso']; b=sp_mag.loc[common]
    res.append((len(common), np.exp(np.median(np.abs(t-a))), np.exp(np.median(np.abs(t-b))), spearmanr(t,a).correlation, spearmanr(t,b).correlation, r2_score(t,a), r2_score(t,b)))
r=np.array(res).mean(0)
print('\nSPECIES HELD OUT, same %d species, isolate genome vs GEM MAG (median of that species\' MAGs)' % r[0])
print('  median fold error : isolate %.2fx   MAG %.2fx' % (r[1],r[2]))
print('  Spearman          : isolate %.3f   MAG %.3f' % (r[3],r[4]))
print('  R2(log)           : isolate %.3f   MAG %.3f' % (r[5],r[6]))
