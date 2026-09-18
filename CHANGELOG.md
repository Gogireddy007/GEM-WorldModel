# Changelog

## 1.4.0, 2026-09-17

Searched for real growth-rate data beyond gRodon and Madin, specifically for the two known training-data gaps. Ruled out BacDive (no growth-rate field), EGGO (predictions from gRodon itself, not real data), and an unverifiable bioRxiv preprint. Found the real answer already cached locally: Madin et al. 2020's full published dataset has 502 species with real doubling time, genome size, and GC content, more than the ~389 this project's pipeline normally pulls. Found 11 of those landing exactly in the two known gaps, pulled real NCBI data for the 10 that resolved, all 10 turned out to be exact GTDB tree tips once matched against GTDB's own pinned accessions. Benchmarked the resulting 353-species corpus across four seeds against both existing corpora: Spearman was worse in every seed against both, R2 mixed, never a clean win. Even a real, carefully targeted addition did not clearly help. Not adopted as the default corpus, following the standard this check was held to, but kept and documented (`features_sample_gapfilled.csv`, `scripts/add_gap_filling_species.py`) as a genuine, honest result.

## 1.3.1, 2026-09-17

Answered a question this project had been assuming rather than checking: is requiring an exact GTDB tree tip still the right default? Ran the clean three-way comparison that had never actually been done, the 175-species exact-tip-only corpus against the 304-species genus-approximation corpus against the 343-species family-approximation corpus, same four seeds, same benchmark, all using the current best model (raw features, gradient-boosted trees). Genus-level approximation wins cleanly, better R2 and Spearman in all four seeds with no exceptions, which reverses an earlier finding in FINDINGS.md that was true for the old fine-tuned JEPA pipeline but not for the model that matters now. Family-level approximation stays a net negative, confirming the two are not equally trustworthy. Updated recommendation: genus-level relaxation should be the default going forward, exact-tip-only should not be, and family-level relaxation should not be used without a better reason than a species simply being available. FINDINGS.md and ACCURACY_PLAN.md updated.

## 1.3.0, 2026-09-17

Finished Phase 3 of the accuracy improvement plan: mapped the training corpus's thin spots properly and tried to patch them with real data. A finer grid scan found ten thin regions, not just the two known before, and counting real GEM catalog genomes in each showed two actually matter: the known thermophile-heavy region (3.0% of the whole catalog) and a newly found Vibrio-heavy region (1.7%), which biases predictions too fast the way the thermophile region biases them too slow, confirmed directly against real predictions (median 0.46h there versus 0.60h elsewhere). Found 92 real labeled species missing from the corpus only because they lacked a genus-level tree match, added a family-level fallback (`features/phylogeny.py:taxonomic_centroid_embeddings`) and pulled real data for the 39 addable ones. Checked before running anything further: none of the 39 land in either known gap. Ran the benchmark anyway across four seeds and it came out flat or worse in every one, never better. Real, negative, honestly reported: the fix for these two gaps needs species that specifically land in them, and nothing reachable from gRodon or Madin does. FINDINGS.md updated with the full gap map and this result.

## 1.2.1, 2026-09-17

Attempted Phase 2 of the accuracy improvement plan, a confidence flag for when a genome sits far from anything the model was trained on, and it did not work. Added `eval/confidence.py`, nearest-neighbor distance in raw feature space, with a threshold set from the training corpus's own internal spread. It looked correct on the known thermophile-bias problem genomes, all ten got flagged. But checked against the whole 7,873-genome deployment set, every single genome got flagged, the threshold was too tight to mean anything at that scale, and the ten known problem genomes turned out not to be unusual at all relative to the rest of the set, nine of them sit below the median distance. Likely a distance-concentration problem from having many features and a small training set. Reported honestly as a real attempt that did not pan out rather than quietly dropped. Not added to any prediction output, since a confidence column that does not discriminate anything is worse than no column at all.

## 1.2.0, 2026-09-17

Finished Phase 0.5 and Phase 1 of the accuracy improvement plan. Traced the fast-growth regime's long-standing poor performance to a real, specific cause: 72% of fast growers get predicted slower than they really are, because a model minimizing squared error over the whole corpus's skewed doubling-time range naturally gets pulled toward the slower majority. Also found that gradient-boosted trees' own feature importance partially agrees with the earlier necessity/sufficiency finding once measured within the slow-growth regime specifically, rather than averaged over the whole corpus, 16S importance rises from 0.24 to 0.41 there, the same direction as before, just not the same top rank. Built a properly cross-validated blend of the new best model (raw features, gradient-boosted trees) with gRodon and Phydon, weights fit honestly with no test-fold leakage. Checked across all seven seeds: the blend never beats gradient-boosted trees alone on overall R2, but it wins Spearman in every seed and substantially tempers the fast-growth regime's worst failures every time. FINDINGS.md updated with both results, kept side by side since which one is "better" depends on the goal.

## 1.1.0, 2026-09-17

Started the accuracy improvement phase tracked in the new ACCURACY_PLAN.md and found the biggest result in the project so far on the first day of it. Checked whether the growth-rate head was leaving accuracy on the table, separate from the encoder, by swapping the small fine-tuned neural network head for gradient-boosted trees. The result did not hold up as an encoder story, once tested against the proper control, raw features with no encoder at all, fed into gradient-boosted trees, beat every encoder-based version tested. Run the full way, with the fast and slow regime breakdown and against both established tools, this new approach beats gRodon outright on R2 and Spearman in all seven of the project's established seeds, and beats Phydon on R2 in all seven seeds too. This is now the most accurate model this project has produced. Two things it does not fix: the fast-growth regime is still very poor, and the tree model's own feature importance does not fully agree with the necessity and sufficiency finding that 16S dominates, both flagged as open follow-ups. FINDINGS.md updated with a new section covering this, the earlier JEPA-versus-raw-features conclusion is kept as written for the setup it was tested under but is no longer the most accurate result available.

## 1.0.0, 2026-09-17

Marking this point as version 1.0, the first stable baseline of the project. Everything up through here is done: the data pipeline is real and reproducible, the JEPA architecture is built and tested, the seed robustness of every headline claim has been checked across seven seeds, the labeled corpus has been grown as far as real data currently allows, and the model has been run end to end on a real deployment target, predicting growth rate for essentially the entire high quality slice of the GEM catalog (9,134 of 9,143 genomes). None of that work is being redone or thrown away.

Starting from this version, the project moves into a new phase with one goal: raise the model's actual predictive accuracy as far as it can go, on any genome it is given, not just add more coverage or more corpus size. That work is being tracked as its own set of tasks from here forward, and every step of it will be written up in this changelog and the research log the same way everything before it was, plainly, with real numbers, and with the outcome reported honestly whether it helps or not.

## 0.10.0, 2026-09-17

Added `scripts/predict_unlabeled_genomes.py`: a genuine end-to-end deployment path, not another benchmark. It fine-tunes one final growth-rate head on the whole 304-species labeled corpus (no held-out fold, this is meant to be used, not scored) and runs it on real GEM MAG genomes that have never had a growth-rate label, producing an actual predicted doubling time for each. Used it to generate predictions for the high-quality (MIMAG HQ) slice of the GEM catalog: cross-referenced our own pulled GEM metadata against the `mimag_quality` field to find 9,143 HQ genomes, then ran real barrnap 16S extraction on the ones still missing it (3,419 genomes, targeted rather than sweeping the whole backlog) and produced predictions for 7,873 of them, 86% of the HQ set. The rest genuinely lack a detectable 16S sequence, a real gap in those genome fragments, not something more extraction time closes. Also added a `--genome-ids-file` option to `gem_slow_features.py` so a specific subset can be prioritized instead of processing genomes in whatever order they happen to sit in.

## 0.9.0, 2026-09-04

Expanded the labeled corpus from 175 to 304 species by adding 129 species that have real growth-rate and taxonomy data but aren't exact GTDB tree tips, giving them an approximate phylogenetic embedding instead (the centroid of same-genus tree-tip embeddings, `features/phylogeny.py:genus_centroid_embeddings`). Every other feature is real, freshly pulled from NCBI. Added `scripts/build_features_expanded.py` and a `--features-file` override on every pipeline script so this never disturbs the original 175-species results. Reran the full pipeline against the 304-species corpus: mixed result, R2 improved modestly across every model but Spearman rank correlation dropped for this project's model while jumping sharply for gRodon/Phydon (likely because the new species skew toward well-studied lab organisms those tools already handle well), so the competitive gap widened rather than closed. What held up and got stronger: the encoder still clearly beats a raw-feature baseline, and the 16S branch's necessity for slow growers widened further. FINDINGS.md updated with a new "Does a larger labeled corpus change anything?" section reporting this honestly, including the parts that didn't improve. Also fixed an overnight sleep stall in the separately-running full-scale 16S extraction job by attaching `caffeinate` to it.

## 0.8.0, 2026-09-03

Extended the seed-robustness check from 2 seeds to 7 (42, 1, 7, 13, 99, 123, 2024), retraining both checkpoints and rerunning the full evaluation suite for each. This turns last version's single-alternate-seed observation into a real replication rate: "full-corpus pretraining beats labeled-only" holds in only 3-4 of 7 seeds depending on the metric, a coin flip, not a finding, and is now reported as an open question rather than a result. What does hold up at a trustworthy rate: pretraining beats the raw-feature baseline (7/7 on R2), and the 16S branch is the most necessary one for slow growers (6/7, the one exception a near-tie). FINDINGS.md's seed-robustness section rewritten with the full 7-seed table and exact fractions instead of a single comparison. No source code changed, this version is checkpoints, evaluation runs, and honest reporting only.

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
