# Changelog

## 0.7.0, 2026-09-02

Added `--seed` overrides to every pipeline script and used them to check whether the project's findings hold under a different random seed. They partly don't: the headline "full-corpus pretraining beats labeled-only pretraining" benchmark result reverses under seed 1 (labeled-only wins on both R2 and Spearman instead), and the necessity/sufficiency absolute R2 numbers swing so much between seeds that one of them flips sign. What does hold under both seeds: pretraining beats a raw-feature baseline with no encoder, the probing comparison favors full-corpus pretraining, and the rrna16s branch is the most necessary one in both the whole-corpus and slow-grower regimes. FINDINGS.md rewritten with a "Seed robustness check" section and the overclaimed parts walked back rather than left standing.

## 0.6.1, 2026-09-02

Added `training/raw_baseline.py` and `scripts/raw_baseline_benchmark.py`: a raw-feature-only baseline (no JEPA encoder, standardized branch features straight into a regression head) evaluated on the identical cross-validation fold membership as the real benchmark, for a genuinely fair check of whether the encoder is earning its place. It is: R2/Spearman improve monotonically from raw features (-0.097/0.318) through labeled-only pretraining (-0.079/0.380) to full-corpus pretraining (-0.055/0.448). FINDINGS.md updated with this result.

## 0.6.0, 2026-09-02

Found and fixed a significant reproducibility bug: nothing in the codebase ever called `.eval()`, so every model's dropout layers were active during every inference call, including the target encoder during pretraining itself (it's never gradient-trained, so its dropout served no purpose, just noise in the training target). `torch.no_grad()` was used correctly everywhere but only disables gradient tracking, not dropout. `TargetEncoder.train()` now overridden to always force eval mode; added `utils/torch_utils.py:eval_mode` context manager and applied it to every genuine inference-only forward pass across `models/jepa.py`, `training/finetune.py`, and `eval/necessity_sufficiency.py`. Both pretrained checkpoints retrained from scratch (the bug affected pretraining itself, not just evaluation) and the full benchmark/probing/necessity-sufficiency suite rerun. The core finding survived and strengthened: full-corpus R2 improved 0.108 to 0.139, the 16S branch's necessity effect for slow growers strengthened from 1.311 to 1.492. FINDINGS.md rewritten with the corrected, now-reproducible numbers. Added 7 regression tests.

## 0.5.1, 2026-09-02

Fixed the last uncross-validated evaluation in the pipeline: `eval/probing.py`'s `linear_probe`/`nonlinear_probe` used a single 70/30 split (n=124 real-labeled species leaves ~37 test samples). Added `linear_probe_cv`/`nonlinear_probe_cv` and made `most_predictive_latent_dim` cross-validated too, `probe_intervene.py` uses these by default now. The corrected numbers actually sharpened the finding: labeled-only pretraining's linear probe accuracy dropped from a lucky 0.789 to a real 0.710, while the full-corpus checkpoint held steady and its AUC improved, making the "full-corpus pretraining helps" result more robust, not less. Added tests for the edge cases (too few samples in the minority class, k auto-shrinking). FINDINGS.md updated with the corrected numbers.

## 0.5.0, 2026-08-31

Moved necessity/sufficiency masking to the same k-fold cross-validation as the benchmark (`eval/necessity_sufficiency.py:necessity_sufficiency_report_cv`), it was evaluating on data the model was fine-tuned on before. Fixed a real bug found while wiring this in (a coverage check against the wrong subset that would have crashed on any regime-restricted call). Added `FINDINGS.md`: the actual synthesis of everything the pipeline has produced, an honest answer to what the project set out to determine, not just per-stage numbers.

## 0.4.0, 2026-08-31

Replaced the genome-derived oligotroph/copiotroph heuristic with a real, genome-independent label. Added `data/madin_traits.py` to pull Madin et al. 2020's curated phenotype trait database (isolation_source, metabolism, and more, real physiological records, not computed from the genome) and `features/ecological_traits.py` to build a trophic label from isolation_source, following the standard nutrient-availability basis in the literature. Covers 124/175 labeled species; ambiguous sources (soil, sediment) are left unlabeled rather than guessed. The old heuristic only agreed with the real label 58.9% of the time and is now a documented `--use-heuristic` fallback in `probe_intervene.py`, not the default. Reran probing: real, above-chance results (0.789-0.842 accuracy depending on checkpoint), a third independent confirmation that full-corpus pretraining beats labeled-only.

## 0.3.2, 2026-08-31

Recovered the 29 labeled species (of 175) that never got CUB/GC/rRNA data in the original run, transient NCBI rate-limiting during that batch, not a permanent gap. All 29 retried cleanly. The labeled corpus is now 175/175 complete on every real feature instead of 146/175. Regenerated both pretrained checkpoints and reran the cross-validated benchmark on the complete data, gRodon and Phydon no longer have to skip rows either, since they also depend on CUB.

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
