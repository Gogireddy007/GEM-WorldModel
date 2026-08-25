# Changelog

## 0.2.0, 2026-08-25

Fixed a handful of real bugs found while running the pipeline at larger scale:

- The pretraining optimizer only covered the context encoder's parameters, so the predictor was never actually updated during training. This is what caused loss to diverge instead of converge once real data was used instead of the short synthetic smoke tests. Fixed, with gradient clipping and an explosion monitor added as well.
- The GTDB tree-placement check was actually checking taxonomy-table membership, which is a much bigger set than the genomes that are real tips on the tree. Real coverage turned out to be 271 accessions (175 species) out of roughly 87,000, not the 93% originally reported. `consolidate.py` and `gtdb.py` now check actual tree membership.
- A truthy-check bug (`nan or 0` evaluates to `nan` in Python) was silently zeroing out rRNA counts even when barrnap ran correctly.
- The gRodon baseline and the benchmark code both crashed on missing CUB values instead of handling them the way gRodon itself would: no CUB, no prediction for that row.
- GEM's genome IDs are JGI-style identifiers, not NCBI accessions, so they never match the official GTDB tree. Added `data/gem_tree.py` to use GEM's own tree instead, matched by OTU id.
- Retry logic with backoff and a shared connection pool for the GEM download scripts, after measuring real failure rates in the 25-70% range under concurrent load against the NERSC portal.

With these fixes, Weeks 1 through 7 now run cleanly on the real, corrected 175-species labeled corpus, and GC content is filled in for basically the entire 52,515-genome GEM corpus (99.95%).

## 0.1.0, 2026-08-24

Initial build covering Weeks 1 through 7 of the plan.

Week 1 pulls data live from gRodon2/Madin on GitHub, GTDB, and the NERSC GEM portal. Week 2 does feature engineering: genome size and GC content computed exactly, CUB via a from-scratch MILC reimplementation, GTDB-distance embeddings via classical MDS over patristic distances, a 16S baseline via NCBI fetch plus k-mer distance, and Arrhenius temperature correction for the growth-rate target. Week 3 is the JEPA core itself, context and target encoders, EMA, branch masking, predictor, latent loss, with unit tests confirming the EMA formula and that gradients never reach the target encoder. Week 4 is the self-supervised pretraining loop with collapse monitoring. Week 5 fine-tunes the growth-rate head and benchmarks against gRodon/Phydon reproductions. Week 6 does linear and nonlinear probing for oligotroph/copiotroph status plus activation intervention. Week 7 does necessity/sufficiency masking per branch, split by doubling-time regime.

The first live runs used a small 20-species sample just to prove the pipeline worked end to end, not to produce meaningful numbers.
