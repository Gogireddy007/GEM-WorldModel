# Changelog

## 0.3.1, 2026-08-28

Replaced the single 70/15/15 train/val/test split in `finetune_benchmark.py` with 5-fold stratified cross-validation (`training/finetune.py:cross_validate`). At n=175 the old split left a 27-sample test set; now every species is fine-tuned from a fresh checkpoint and held out exactly once, so the benchmark covers all 175 species instead of a fragile slice. gRodon/Phydon baselines are refit per fold on the same splits for a fair comparison. `--checkpoint` flag added so the script can benchmark either the labeled-only or full-corpus pretrained model.

## 0.3.0, 2026-08-28

First real run of combined pretraining across the labeled corpus and the full GEM MAG corpus (`pretrain_full.py`). Two scale bugs came up and got fixed:

- The 16S k-mer distance computation was an O(n²) Python loop, fine at 175 genomes, hung for minutes at GEM's 1,883 real-16S genomes. Rewrote it as a vectorized matrix operation with identical math, 0.43 seconds instead of hanging.
- Training loss climbed steadily instead of converging once the corpus was actually large (about 155,000 optimizer steps over 200 epochs versus ~600 in earlier runs), a scale-drift failure mode that plain MSE on unnormalized latents is prone to and that gradient clipping alone doesn't stop. Fixed by L2-normalizing both sides of the loss (the SimSiam/BYOL trick), which converges cleanly now.

Also wired the GEM corpus's real 16S data (1,875 genomes) into pretraining as a proper 3-branch sub-corpus, alongside a 2-branch sub-corpus for the other ~49,700 GEM genomes and the 3-branch labeled corpus, all training one shared model.

## 0.2.1, 2026-08-26

The two long-running GEM corpus downloads finished. GC content now covers 52,489 of 52,515 genomes (99.95%). Real 16S extraction via barrnap completed its full 5,000-genome run: 4,652 genomes processed successfully (7% error rate), 1,883 with a real 16S sequence recovered. Combined with the metadata-derived genome size, rRNA/tRNA counts, and phylogenetic embeddings that already covered the full corpus, this is now the actual, final state of the unlabeled GEM dataset for this build, not a partial or projected one.

## 0.2.0, 2026-08-25

Fixed a handful of real bugs found while running the pipeline at larger scale:

- The pretraining optimizer only covered the context encoder's parameters, so the predictor was never actually updated during training. This is what caused loss to diverge instead of converge once real data was used instead of the short synthetic smoke tests. Fixed, with gradient clipping and an explosion monitor added as well.
- The GTDB tree-placement check was actually checking taxonomy-table membership, which is a much bigger set than the genomes that are real tips on the tree. Real coverage turned out to be 271 accessions (175 species) out of roughly 87,000, not the 93% originally reported. `consolidate.py` and `gtdb.py` now check actual tree membership.
- A truthy-check bug (`nan or 0` evaluates to `nan` in Python) was silently zeroing out rRNA counts even when barrnap ran correctly.
- The gRodon baseline and the benchmark code both crashed on missing CUB values instead of handling them the way gRodon itself would: no CUB, no prediction for that row.
- GEM's genome IDs are JGI-style identifiers, not NCBI accessions, so they never match the official GTDB tree. Added `data/gem_tree.py` to use GEM's own tree instead, matched by OTU id.
- Retry logic with backoff and a shared connection pool for the GEM download scripts, after measuring real failure rates in the 25-70% range under concurrent load against the NERSC portal.

With these fixes, the full pipeline now runs cleanly on the real, corrected 175-species labeled corpus, and GC content is filled in for basically the entire 52,515-genome GEM corpus (99.95%).

## 0.1.0, 2026-08-24

Initial build covering the full pipeline: data acquisition, feature engineering, the JEPA architecture, pretraining, fine-tuning, and explainability analysis.

Data acquisition pulls data live from gRodon2/Madin on GitHub, GTDB, and the NERSC GEM portal. Feature engineering computes genome size and GC content exactly, CUB via a from-scratch MILC reimplementation, GTDB-distance embeddings via classical MDS over patristic distances, a 16S baseline via NCBI fetch plus k-mer distance, and Arrhenius temperature correction for the growth-rate target. The JEPA core itself has context and target encoders, EMA, branch masking, a predictor, and a latent loss, with unit tests confirming the EMA formula and that gradients never reach the target encoder. Pretraining is the self-supervised loop with collapse monitoring. Fine-tuning attaches and trains the growth-rate head, then benchmarks against gRodon/Phydon reproductions. Probing does linear and nonlinear analysis of oligotroph/copiotroph status plus activation intervention. Necessity/sufficiency analysis masks each branch, split by doubling-time regime.

The first live runs used a small 20-species sample just to prove the pipeline worked end to end, not to produce meaningful numbers.
