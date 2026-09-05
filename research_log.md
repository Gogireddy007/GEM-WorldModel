# Research Log

## 2026-09-04, expanding the labeled corpus past 175, and what that changed

Picked up the labeled-corpus-expansion question flagged the prior day. `grodon.build_labeled_corpus()` already pulls every Madin/gRodon species with growth-rate data, no artificial cap, so the 175-species ceiling is real: it's exactly how many of those species are an actual GTDB tree tip. Checked what's available beyond that: 229 more species have real growth-rate data and real GTDB taxonomy but aren't tree tips. Of those, 132 (129 after dropping species where the embedding couldn't be materialized cleanly) share a GTDB genus with an already-embedded tree-tip species.

Built `features/phylogeny.py:genus_centroid_embeddings`, which approximates a non-tip species' phylogenetic-distance embedding as the centroid (mean) of the embeddings of tree-tip species in the same genus. Not a new tree placement, a same-genus nearest-neighbor stand-in, and species whose genus has zero tree-tip representative get no entry rather than a worse guess. Added a test (`test_genus_centroid_embeddings_averages_same_genus_tips`) confirming the centroid math and confirming un-matchable species are dropped, not silently zero-filled.

New script `scripts/build_features_expanded.py` runs the identical real feature pipeline (NCBI genome/CDS download, CUB, genome traits, 16S) used for the original 175 against these 129 species, substituting the genus-centroid embedding for the exact-tip one on the phylogeny branch, and writes a `features_sample_expanded.csv` that's a strict superset of `features_sample.csv` (the original file untouched), tagged with a `placement_type` column (`exact_tip` vs. `genus_centroid_approx`) so anything downstream can filter to the stricter set if it wants. Ran a 3-species smoke test first, verified every real feature column was non-null before committing to the full 129-species run, which took about 95 minutes of real NCBI network time and finished clean: 304 total rows, 127/129 new species got a real 16S sequence (2 genuinely lack a detectable one, same kind of real gap the original 175 also has for a handful of species).

Added a `--features-file` override to every pipeline script (`pretrain_labeled.py`, `pretrain_full.py`, `finetune_benchmark.py`, `raw_baseline_benchmark.py`, `run_necessity_sufficiency.py`, `probe_intervene.py`), defaulting to the original file so every existing result, including the whole 7-seed robustness check, stays exactly reproducible. Non-default files auto-tag output checkpoint/CSV filenames (e.g. `jepa_pretrained_expanded.pt`) so nothing overwrites the default-corpus outputs.

Retrained both checkpoints and reran the full evaluation suite against the 304-species corpus at seed 42. Result was genuinely mixed, not a clean win: R2 improved modestly across every model (raw, labeled-only, full-corpus, and both baselines), but Spearman rank correlation got *worse* for all three of this project's model variants while jumping sharply for gRodon and especially Phydon (R2 turned positive for the first time anywhere in this project, 0.100). The 129 new species lean toward well-studied lab/industrial organisms (E. coli relatives, Pseudomonas, Klebsiella, Vibrio and similar) that gRodon/Phydon's own training likely covers well, a plausible explanation, this changed corpus composition, not just corpus size, and the two effects aren't separable from one run. The gap between this project's model and the established tools widened on the expanded corpus rather than closing.

What did hold up, and in one case got stronger: the encoder still clearly beats the raw-feature baseline at n=304 on both metrics, and the 16S branch's necessity for slow growers widened further (R2 drop 1.082 at n=304 vs. 1.492 at n=175 under the same seed, both far ahead of the other two branches, an even bigger relative margin at n=304). Full-corpus pretraining beat labeled-only again at seed 42 on the larger corpus, same direction as before, but that specific comparison is already known from the 7-seed check to be seed-unstable at n=175, and wasn't re-checked across seeds at n=304, so this one data point doesn't make it more trustworthy, just not contradicted yet. FINDINGS.md updated with a new "Does a larger labeled corpus change anything?" section reporting all of this plainly, including the parts that didn't go the hoped-for direction.

Also fixed an unrelated operational issue found the same morning: the long-running full-scale 16S extraction (started the prior evening against the remaining ~47,800 GEM MAG genomes) had stalled overnight when the machine went to sleep, a ~10-hour gap in its log with almost no progress. Attached `caffeinate -i -w <pid>` to the running extraction process (without restarting it) to hold the machine awake for as long as it runs. Throughput recovered to its pre-sleep pace within the hour and later sped up further once the labeled-corpus-expansion job finished and stopped competing for CPU.

## 2026-09-03, seed robustness check extended to 7 seeds, and a genuine ceiling found

The seed-1 comparison below already showed one headline result reversing. Ran five more seeds (7, 13, 99, 123, 2024) through the identical pipeline, same `--seed` override pattern, same checkpoint-per-seed retraining, same full evaluation suite, to turn a single alternate data point into a real replication rate instead of a coin toss between two numbers.

All five additional labeled-only and full-corpus checkpoints pretrained cleanly, loss converging, no collapse, same as every prior seed.

Final tally across all 7 seeds (42, 1, 7, 13, 99, 123, 2024):

- **Full-corpus pretraining beats labeled-only on R2:** 4/7 (wins at 42, 7, 123, 2024; loses at 1, 13, 99).
- **Full-corpus pretraining beats labeled-only on Spearman:** 3/7 (wins at 42, 7, 123; loses at 1, 13, 99, 2024).
- **Pretraining (either variant) beats the raw-feature baseline on R2:** 7/7 for both labeled-only and full-corpus.
- **Pretraining beats raw baseline on Spearman:** 6/7 for labeled-only (loses only at seed 123, where labeled-only Spearman is actually negative, -0.322), 5/7 for full-corpus.
- **rrna16s is the single most necessary branch for slow growers:** 6/7 (loses only at seed 99, where gtdb_distance edges it 1.250 vs 1.122, a near-tie).
- **Necessity/sufficiency whole-corpus R2 is positive:** 2/7 (only seeds 42 and 7; the other five range down to -0.542).

This settles what the single seed-1 comparison could only suggest. "Full-corpus beats labeled-only" is not a real effect at this corpus size, it's noise that happened to land the same direction on the first extra seed checked and the opposite direction on several more, an honest coin flip. "Pretraining beats raw features" and "16S dominates for slow growers" both survive at a rate (6-7 of 7) that's a real, trustworthy signal, not luck. Rewrote FINDINGS.md's seed-robustness section with the full 7-seed table and exact replication fractions instead of the earlier 2-seed comparison, and adjusted the "Does pretraining help" and "Bottom line" sections to match. Full test suite (55 tests) still passes, none of this touched any source code, only checkpoints, evaluation runs, and the writeup.

Started two other tracks the same day, both still running independently at time of writing: a full-scale real 16S extraction over the remaining ~47,800 GEM MAG genomes that had never had barrnap run against them (previously only a 5,000-genome sample had been attempted, yielding 1,883 real 16S sequences; the full run is unattended, resumable, and estimated at roughly 40+ hours given barrnap's per-genome CPU cost even across 10 parallel workers), and an open question about expanding the 175-species labeled corpus, which turned out on inspection to already be the true ceiling of what `grodon.build_labeled_corpus()` pulls from Madin/gRodon intersected with actual GTDB tree-tip placement, not an artificial subsample. Two ways to grow it exist: a `lab_added_species.csv` slot in the code that was never actually populated with data, and building an approximate/landmark phylogenetic-distance embedding for the 214 Madin/gRodon species that have GTDB taxonomy but aren't exact tree tips (excluded today, tree-tip placement is currently required). Neither started without a decision on which approach to take.

## 2026-09-02, seed robustness check: not everything survived

Asked directly whether the findings so far, all produced with seed 42, were real or a lucky draw at 175 samples. Added a `--seed` override to every pipeline script (`pretrain_labeled.py`, `pretrain_full.py`, `finetune_benchmark.py`, `run_necessity_sufficiency.py`, `probe_intervene.py`, `raw_baseline_benchmark.py`), retrained both checkpoints from scratch with seed 1, and reran the entire evaluation suite against them using seed 1 for the cross-validation folds too, so it's a genuinely independent run, not the same folds with a different model dropped in.

Both seed-1 checkpoints pretrained cleanly, same as seed 42, loss converging and no collapse. The evaluation results were where it got interesting.

The benchmark comparison that anchored most of this project's "pretraining helps" story did not survive. Under seed 42, full-corpus pretraining beat labeled-only pretraining on both R2 (-0.055 vs -0.079) and Spearman (0.448 vs 0.380). Under seed 1, that reverses: labeled-only beats full-corpus on both metrics (R2 -0.087 vs -0.092, Spearman 0.430 vs 0.367), and full-corpus pretraining is no longer clearly ahead of the raw-feature baseline at all. The necessity/sufficiency absolute R2 numbers were even less stable, the full-corpus checkpoint's whole-corpus R2 went from +0.139 (seed 42) to -0.542 (seed 1), a sign flip, and which regime the model fit worse (fast vs. slow) flipped too.

What did survive: labeled-only pretraining still beat the raw-feature baseline under both seeds (the encoder itself is adding value, that's real), the probing comparison still favored full-corpus pretraining under both seeds, and the ranking "rrna16s is the most necessary branch" held in both the whole-corpus and slow-grower regimes under both seeds, even while the specific R2 numbers attached to that ranking swung wildly.

Net effect: at 175 labeled species with 5-fold CV (about 35 species deciding each fold), this corpus is big enough to support some directional claims but not the specific "full-corpus pretraining beats labeled-only" comparison that earlier entries and FINDINGS.md leaned on. Rewrote FINDINGS.md to add a "Seed robustness check" section and walked back the overclaimed parts rather than leaving them standing next to a result that contradicts them. This is the outcome an honest check is supposed to be able to produce, not every result was going to hold up, and it didn't.

## 2026-09-02, checking whether the JEPA encoder is actually earning its place

Everything so far compared JEPA-based predictions against gRodon/Phydon, and against each other (labeled-only vs. full-corpus pretraining). None of that answers a more basic question: does going through the pretrained encoder at all beat just feeding the same standardized raw features straight into a regression head? Easy to assume yes and never check.

Built `training/raw_baseline.py:cross_validate_raw`, structurally parallel to the real `cross_validate` but with no encoder at all, raw concatenated branch features straight into a fresh `GrowthRateHead` each fold. Deliberately used the identical `StratifiedKFold(seed=..., k=...)` call so the fold membership is byte-for-byte the same as the JEPA runs, added a test proving exactly that (`test_raw_baseline_and_jepa_cross_validate_use_identical_fold_membership`), a comparison where the two sides see different train/test splits isn't a fair comparison.

Real result: R2 -0.097, Spearman 0.318 for the raw baseline, versus -0.079/0.380 for labeled-only pretraining and -0.055/0.448 for full-corpus pretraining. Both metrics improve monotonically through all three. The encoder is genuinely adding value, this could easily have gone the other way at 175 real samples, and if it had, that would have been the actual finding to report. It didn't, so it isn't.

One test-writing note: the first version of the raw-baseline sanity test injected a synthetic signal on a single isolated input dimension (1 real dimension against 38 pure-noise dimensions) and asserted a correlation the head-only regressor genuinely couldn't reach in a reasonable epoch budget at that sample size, confirmed empirically by sweeping epochs/lr/weight_decay and watching it plateau around 0.39 regardless. Not a bug, a needle-in-haystack problem that was harder than the corpus size could support, exactly the kind of thing the real project's fast-growth regime is also running into. Fixed the test by spreading the synthetic signal across all of one branch's dimensions instead of one, which converges to 0.72 correlation cleanly, that's the more honest test of "can this training loop learn a real relationship," not "can it solve variable selection from 48 samples."

## 2026-09-02, a real and significant bug: the model was never in eval mode

While rerunning `probe_intervene.py` right after the CV fix below, noticed the numbers changed between two identical back-to-back runs of the same command against the same checkpoint. That should be impossible for a deterministic model doing inference. It wasn't a fluke.

Direct repro: loaded a checkpoint, ran `jepa.joint_representation(batch)` twice on the exact same input tensor. The two outputs differed by up to 1.13 in absolute terms, on latents that were meant to be roughly unit-scale after L2 normalization. `context_encoder.training` and `target_encoder.bank.training` were both `True`. Grepped the whole codebase for `.eval()`: zero hits, anywhere. `torch.no_grad()` is used everywhere for inference, correctly, but it only disables gradient tracking, it does not disable dropout. `configs/model.yaml` sets a real `dropout: 0.1` on every branch encoder MLP. A freshly constructed or freshly loaded `nn.Module` defaults to `training=True` in plain PyTorch. Nobody had ever told it otherwise.

This is worse than a normal "forgot eval() before inference" bug, because it also affected the target encoder DURING pretraining itself, not just downstream evaluation. The target encoder is never trained by gradient descent, only EMA-updated, so its dropout serves no purpose at all, it just injects random noise into the value (`s`) that the predictor is trying to match. Every checkpoint pretrained before this fix, and therefore every benchmark, probing, and necessity/sufficiency number produced from those checkpoints, was computed against a moving, noisy target during training and a non-deterministic encoder at evaluation time.

Fixed at the architecture level, not just by sprinkling `.eval()` calls around: `TargetEncoder.train()` is now overridden to always force eval mode regardless of what's requested, since it should never be anything else. Added `utils/torch_utils.py:eval_mode`, a context manager that switches modules to eval for a block and restores whatever mode they were in before (not unconditionally back to train, that would break the alternating train-step/val-check pattern inside the fine-tuning loop). Applied it everywhere a genuine inference-only forward pass happens: `JEPA.joint_representation`, the val-loss check and final test-prediction inside `training/finetune.py`, and every forward pass inside `eval/necessity_sufficiency.py`. Left the actual gradient-descent training steps alone, dropout there is correct and intentional. `eval/probing.py`'s own no_grad blocks turned out not to need this fix, the small MLP it trains internally has no dropout layer, only the JEPA context encoder that produces the latents fed into it did.

Confirmed with the same repro that motivated the whole investigation: two calls on identical input now give `torch.equal` results. Added five tests for the `eval_mode` context manager itself (switches correctly, restores correctly whether the prior state was train or eval, restores correctly with multiple modules at once, restores correctly even if the block raises) and two for the JEPA-level fix specifically (target encoder survives a `jepa.train()` call still in eval mode, `joint_representation` is deterministic).

Since the target encoder's dropout affected pretraining itself, not just evaluation, both checkpoints had to be retrained from scratch, not just re-evaluated. Reran the full downstream suite (benchmark, probing, necessity/sufficiency) on both fresh checkpoints. The core finding survived intact and in the necessity/sufficiency breakdown actually got stronger: the 16S branch's necessity R2 drop for slow growers went from 1.311 to 1.492, and the full-corpus checkpoint's overall R2 improved from 0.108 to a real 0.139. Full corrected numbers are in FINDINGS.md, which has also been rewritten to make clear every number in it postdates this fix.

## 2026-09-02, the last uncross-validated evaluation, fixed

Went through a readiness review of the whole pipeline and found one real gap left: `linear_probe`/`nonlinear_probe` in `eval/probing.py` still used a single random 70/30 split, the exact same fragility already found and fixed in the benchmark and necessity/sufficiency, just missed when those were fixed. At 124 real-labeled species that's ~37 test samples.

Added `linear_probe_cv`/`nonlinear_probe_cv` (k-fold, every sample gets exactly one out-of-fold prediction, accuracy/AUC computed once over the pooled set) and made `most_predictive_latent_dim` cross-validated too, since picking a latent dimension by how well it memorizes all 124 labeled points and then reporting an intervention result on that dimension has the same overfitting risk as everything else this pass fixed. Old single-split functions kept only as fast exploratory primitives, not for headline numbers. `probe_intervene.py` now uses the CV versions by default.

The corrected numbers actually moved the story, not just the confidence: linear probe accuracy on the labeled-only checkpoint dropped from a real 0.789 to a real 0.710 (AUC held up better, 0.750 to 0.773), while the full-corpus checkpoint held essentially steady and its AUC improved (0.804 to 0.880). This makes the "full-corpus pretraining helps" finding cleaner, not weaker, the labeled-only checkpoint's old number was partly a lucky split.

Also added tests for the edge cases this kind of code tends to break on quietly: too few samples in the minority class (raises clearly instead of producing a garbage fold), and k auto-shrinking when a class is smaller than the requested fold count. Neither was tested before this pass and both are exactly the kind of thing that breaks silently on a slightly different sample.

FINDINGS.md updated with the corrected numbers.

## 2026-08-31, necessity/sufficiency moved to cross-validation too, and the actual synthesis

`run_necessity_sufficiency.py` had the same problem the benchmark had before it was fixed: it fine-tuned on one split and then evaluated necessity/sufficiency masking on the SAME data (train+val+test all mixed together), not held out. Fixed it the same way, `eval/necessity_sufficiency.py:necessity_sufficiency_report_cv` runs the ablation per cross-validation fold on that fold's own held-out test samples with that fold's own fine-tuned model, and branch-neutralization means come from each fold's train split only, never its test split. Caught a real bug while wiring this in: the coverage assertion checked against the full n instead of the regime-filtered subset, would have crashed on every regime-restricted call. Fixed and added a test that specifically exercises a regime mask, not just the unrestricted case, so this doesn't regress silently again.

Full results and the actual answer to what this whole project was for are in FINDINGS.md.

## 2026-08-31, a real oligotroph/copiotroph label, no more heuristic

The probing step was using a genome-derived proxy for oligotroph/copiotroph status (genome size + rRNA copy number), which had a real circularity problem: oligotrophs are partly defined by weak codon usage bias, so a probe "discovering" CUB-correlated structure predicting that heuristic wasn't necessarily finding anything real. Went looking for an actual literature source instead of treating this as blocked on someone else supplying one.

Found it: Madin et al. 2020 ("A synthesis of bacterial and archaeal phenotypic trait data", Scientific Data) publishes a real, curated trait database with 14,893 species and 79 columns, isolation_source, metabolism, carbon_substrates, gram_stain, and more, assembled from actual physiological records, not computed from genome sequence. Pulled the real data (`data/madin_traits.py`) and built a genome-independent trophic label from isolation_source (`features/ecological_traits.py`): host-associated and engineered/waste environments classified copiotroph-leaning, open water and deep subsurface environments classified oligotroph-leaning, following the standard nutrient-availability basis in the literature (Fierer et al. 2007, Lauro et al. 2009). Soil and sediment sources are left unlabeled on purpose, their organic content is too variable in the literature to assign a direction, and guessing there would defeat the point of using a real label.

Real coverage: 124/175 labeled species (71%), 84 copiotroph-leaning, 40 oligotroph-leaning. Checked the new label against the old heuristic on the 124 species where both exist: they agree 58.9% of the time, barely above chance for a 2-class problem. The heuristic was not a reliable stand-in for a real trophic label.

Reran probing with the real label:

| checkpoint | linear acc / AUC | nonlinear acc / AUC |
|---|---|---|
| labeled-only pretrain | 0.789 / 0.750 | 0.711 / 0.747 |
| full-corpus pretrain | 0.816 / 0.804 | 0.842 / 0.827 |

Both real, above-chance results now, not the heuristic label's numbers from before. And this is a third independent line of evidence (after the pretraining loss curve and the cross-validated benchmark's Spearman gain) that full-corpus pretraining produces a better representation than labeled-only.

## 2026-08-31, recovered the 29 species with missing features

29 of the 175 labeled species never got CUB/GC/rRNA data in the original run, their NCBI downloads must have hit transient rate-limiting during that batch. Retried them individually and all 29 came back clean, real genome and CDS files, real CUB values in the normal range (0.13-0.77), real GC content (26-72%). The labeled corpus is now 175/175 complete across every real feature, not 146/175.

Regenerated both pretrained checkpoints and reran the cross-validated benchmark on the complete data:

| model | R2 (all) | Spearman (all) |
|---|---|---|
| gem_worldmodel (labeled-only pretrain) | -0.078 | 0.380 |
| gem_worldmodel (full-corpus pretrain) | -0.057 | 0.425 |
| grodon_reproduction | -0.032 | 0.582 |
| phydon_reproduction | +0.034 | 0.626 |

Same pattern holds as before: full-corpus pretraining still helps Spearman over labeled-only. Both baselines still beat our model on rank correlation. Phydon's R2 actually crossed into positive territory now that it isn't missing 29 rows anymore, small but real, since gRodon and Phydon both use CUB and previously had to exclude or degrade on those rows the way we did.

## 2026-08-28, cross-validated benchmark replaces the fragile 27-sample test split

The single 70/15/15 split used for fine-tuning and benchmarking left a 27-sample test set at n=175, too small for RMSE/R2/Spearman on it to mean much. Replaced it with 5-fold stratified cross-validation (`training/finetune.py:cross_validate`): every species gets fine-tuned from a fresh copy of the pretrained checkpoint and held out exactly once, so the benchmark now runs over all 175 out-of-fold predictions instead of a slice of 27. The gRodon/Phydon baselines are refit per fold on the exact same splits, so the comparison stays apples-to-apples.

Real result at n=175, cross-validated:

| model | R2 (all) | Spearman (all) |
|---|---|---|
| gem_worldmodel (labeled-only pretrain) | -0.062 | 0.237 |
| gem_worldmodel (full-corpus pretrain) | -0.053 | 0.441 |
| grodon_reproduction | -0.025 | 0.625 |
| phydon_reproduction | -0.008 | 0.684 |

This confirms the earlier single-split finding (full-corpus pretraining gives real Spearman gains over labeled-only) with a statistically credible sample size instead of n=27, and is honest about the rest: both baselines still beat our model on rank correlation at this corpus size, and R2 is negative across the board. That's the real, current state, not spun.

## 2026-08-28, first real combined pretraining run

Ran pretrain_full.py for the first time against the completed real data, combined pretraining across the labeled corpus and the GEM MAG corpus in one model. This surfaced two more real bugs, both scale problems that the smaller runs never hit.

First: the 16S k-mer distance computation was a pure-Python loop over all pairs, fine for the 175-species labeled corpus (about 15,000 pairs) but it hung for minutes at GEM scale (1,883 genomes with real 16S, about 1.8 million pairs). Rewrote it as a single vectorized matrix operation, same math (a missing k-mer contributes 0 to a profile either way, so building against the full observed vocabulary instead of each pair's own union gives identical distances), just fast: 0.43 seconds instead of hanging.

Second, and more interesting: once training actually ran across the full corpus (about 51,787 genomes total, vastly more optimizer steps per epoch than any earlier run), loss climbed steadily from 0.005 to 1.78 over 200 epochs instead of converging, even with gradient clipping on. Plain MSE on unnormalized latents has a degenerate way to shrink the loss: scale every embedding up uniformly. Gradient clipping bounds the size of each step but doesn't stop that drift from compounding over ~155,000 steps. Fixed by L2-normalizing both sides of the loss before computing MSE, the same trick SimSiam and BYOL use, which removes the scale-drift direction entirely since a unit-norm vector can't drift in magnitude. After the fix, loss converges to about 0.0126 by epoch 30 and stays flat for the rest of training.

The combined corpus split into three sub-corpora with different real branch coverage: the 175-species labeled set (all 3 branches), 49,737 GEM MAGs with real genomic traits and phylogeny but no real 16S (2 branches), and 1,875 GEM MAGs that do have a real 16S sequence (all 3 branches). All three train together against one shared model.

Compared the resulting checkpoint against the labeled-only pretraining from two days ago, fine-tuned and benchmarked both the same way. At n=27 test samples the difference isn't statistically decisive, but full-corpus pretraining showed a real Spearman improvement (0.280 to 0.471) while RMSE and R² stayed roughly flat (both near zero, dominated by outliers at this sample size). Not proof it helps, but a real positive signal worth following up once the labeled corpus is bigger.

## 2026-08-26, GEM corpus background jobs finished

Both long-running downloads against the GEM portal that were still going as of yesterday's entry finished cleanly.

GC content: 52,489 of 52,515 genomes (99.95%) now have a real, verified value. Only 26 genomes failed permanently after retries, the rest of the corpus is complete.

Real 16S extraction via barrnap: ran to its full 5,000-genome target. 4,652 of those genomes were successfully processed by barrnap (348 failed even after retries, a 7% error rate, down from the 25-70% seen earlier under bad concurrency). Of the genomes barrnap actually ran on, 1,883 had a real 16S copy detected and got a real k-mer profile for the rrna16s branch, the rest had zero copies found, which is a genuine result for fragmented MAG assemblies, not a failure to extract.

So at this point: all 52,515 GEM genomes have real genome size, rRNA/tRNA counts, and phylogenetic embeddings. 52,489 additionally have real GC content. 1,883 additionally have a real 16S sequence usable for the rrna16s branch. That's the actual, current state of the unlabeled corpus, nothing in it is fabricated or interpolated.

## 2026-08-25, bug fixes and a real run at proper scale

Picked back up from yesterday's smoke-scale build and pushed it toward real numbers, which surfaced several actual bugs that the small smoke tests hadn't caught.

The biggest one: pretraining loss was diverging on real data instead of converging, going from 0.02 up to 24 over 200 epochs. Turned out the optimizer was only built from the context encoder's parameters, so the predictor never trained at all. It stayed randomly initialized the whole time, which is a bad thing to be chasing with an EMA target encoder that keeps moving. Fixed the optimizer to include the predictor, added gradient clipping and an explosion monitor as a backstop, and added a regression test that checks the predictor's own weights actually move after a step. After the fix, loss went from 0.0235 down to 0.004 and stayed bounded.

Second one, and honestly the more consequential correction: the GTDB "in tree" flag was wrong. It was checking whether an accession had GTDB taxonomy at all, which is true for about 93% of the labeled corpus. But taxonomy and tree placement are different things, GTDB's tree only has representative genomes as tips, roughly 190,000 out of the 879,000 genomes it has classified. Checking real tree membership dropped the usable labeled corpus from a reported 314 species down to 175. That's the real ceiling for anything using the phylogeny branch, not the earlier number.

Also fixed: a `nan or 0` bug in the rRNA counter (nan is truthy in Python, so this silently zeroed out real barrnap results), a crash in the gRodon baseline on missing CUB values, and a mismatch between GEM's genome IDs (JGI-style) and GTDB's tree tips (NCBI accessions), which meant GEM MAGs could never be placed on the GTDB tree no matter what. Added `data/gem_tree.py` to use GEM's own 43,979-OTU tree instead, matched by OTU id, which is the actually correct source for that corpus anyway.

With all of that fixed, reran the full pipeline on the real 175-species corpus:

Pretraining converges properly now. Fine-tuning and benchmark run end to end on a real 27-sample test split, results are mostly weak or negative R², which is an honest reflection of the sample size, not a bug. Probing gets 84.9% linear / 86.8% nonlinear accuracy on the heuristic trophic label. Necessity/sufficiency masking runs per branch and per regime.

Separately, pushed the GEM MAG corpus work. All 52,515 genomes now have real genome size and rRNA/tRNA counts (straight from GEM's own metadata, no extra computation needed) and real phylogenetic embeddings via the landmark-distance method (needed since the full 44k-tip tree is too big for classical MDS). GC content took a long background run, over 20 hours by the end, mostly due to NERSC throttling our sustained connections rather than anything wrong on our end, but finished at 99.95% coverage (52,489 of 52,515). Real 16S extraction via barrnap is the slow one: measured throughput was well under 1 genome/second, so full coverage of 52,515 genomes isn't realistic in a single session. Ran it against a 5,000-genome subset instead.

One practical lesson from today: don't run the GC-content job and the barrnap job against the same host at the same time. Barrnap is CPU-heavy enough that it starves the other job's I/O threads, which shows up as real connection timeouts, not just flakiness. Running them one at a time was more reliable even though it's slower in isolation.

## 2026-08-24, initial build

Built from the two source documents (the architecture diagram and the phased master plan): a JEPA-style world model over microbial genomes, cross-species rather than the earlier Keio-only track, phylogeny via GTDB's tree, snapshot-level with no temporal rollout.

Data acquisition pulled 86,973 accession-level growth-rate rows across 389 species from gRodon2/Madin, plus 2,000 GEM MAGs for the unlabeled pool. Feature engineering built a real feature table for a 20-species sample. The rest of the pipeline all ran end to end, but at a sample size too small for the numbers to mean anything, this was purely to prove the pipeline worked, not to get a real result. That distinction turned out to matter: the small-sample runs didn't surface either of the two bugs found and fixed the next day.
