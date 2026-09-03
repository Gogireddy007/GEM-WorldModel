# Findings

What this project set out to answer: what controls microbial growth rate, and does the answer differ for fast-growing organisms (doubling time under 5 hours) versus slow-growing ones (5 hours or more). Everything below is from real runs on real data, cross-validated where the sample size demanded it, current as of 2026-09-02. Numbers this small a corpus produces are not the kind you'd put in a paper's abstract, they're honestly reported anyway because that's the actual state of the evidence right now.

A real bug was found and fixed right before this version: nothing in the codebase ever called `.eval()` on the model, so its dropout layers were active during every inference call, including the target encoder during pretraining itself, where it should never have been active at all (see research_log.md, 2026-09-02). Every checkpoint and every number below was regenerated after the fix. The core finding held up, and in a couple of places got stronger, not weaker.

A follow-up question then had to be asked directly: were those numbers real, or just a lucky seed? Every checkpoint and every evaluation below was rerun end to end with a second random seed to check. The answer was mixed, and it's reported honestly in "Seed robustness check" further down: some of what's claimed here holds up, one of the headline comparisons does not.

## The corpus, briefly

175 species with a real measured growth rate, GTDB tree placement, genomic traits, and 16S data (the true ceiling given how few gRodon/Madin accessions are actual GTDB tree tips, see research_log.md for how that number was arrived at). 51,612 GEM MAGs with real genomic traits and phylogenetic placement, 1,875 of those also with a real 16S sequence, used as an unlabeled pretraining supplement since pretraining doesn't need growth-rate labels.

## Does self-supervised pretraining on the larger unlabeled corpus actually help?

Under the seed this project has used throughout (seed 42), yes, consistently, across three independent measurements:

1. **Benchmark (5-fold cross-validated, n=175):** full-corpus pretraining lifts Spearman rank correlation from 0.380 to 0.448 over labeled-only pretraining. R2 stays roughly flat and negative (-0.079 to -0.055, this is a small, noisy corpus).
2. **Probing (real trophic label, 5-fold cross-validated, n=124):** linear probe accuracy goes from 0.718 to 0.766, AUC from 0.767 to 0.794. Nonlinear probe goes from 0.734 to 0.750, AUC from 0.801 to 0.794 (essentially flat there, the one place this measurement didn't clearly favor full-corpus).
3. **Necessity/sufficiency (5-fold cross-validated):** full-corpus pretraining gets a real positive R2 (0.139) on the whole labeled corpus; labeled-only pretraining doesn't (-0.164).

Three different evaluations, three different metrics, the same direction in five of six numbers under this one seed. But a second seed was run specifically to check whether that agreement was real or a coincidence of this particular random draw, and it wasn't fully real: see "Seed robustness check" below, the benchmark comparison (item 1) reverses under a different seed, while the probing comparison (item 2) and the qualitative necessity/sufficiency ranking (item 3, which branch matters most) hold up.

## Is the JEPA encoder itself earning its place?

This has to be checked directly, not assumed. Ran the raw (standardized) branch features straight into a growth-rate head, no encoder, no pretraining at all, through the exact same k-fold cross-validation (identical fold membership) as everything else:

| model | R2 (all) | Spearman (all) |
|---|---|---|
| raw features, no encoder | -0.097 | 0.318 |
| JEPA, labeled-only pretrain | -0.079 | 0.380 |
| JEPA, full-corpus pretrain | -0.055 | 0.448 |
| gRodon reproduction | -0.032 | 0.582 |
| Phydon reproduction | 0.034 | 0.626 |

Both metrics improve monotonically going from raw features to labeled-only pretraining to full-corpus pretraining, under seed 42. Under seed 1, the first step still holds (labeled-only pretraining beats raw features on both metrics) but the second step doesn't (full-corpus pretraining falls back to roughly raw-baseline territory, see "Seed robustness check"). So the honest claim is narrower than it first looked: pretraining the encoder at all is earning its place, more pretraining data is not reliably earning more, at least not yet at this labeled-corpus size.

## What controls growth rate, by branch

Necessity/sufficiency masking (cross-validated, full-corpus checkpoint) on the whole labeled corpus:

| branch | necessity (R2 drop if removed) | sufficiency (R2 alone) |
|---|---|---|
| rrna16s (16S phylogeny) | 0.473 | -0.029 |
| genomic_traits (CUB, GC, genome size, etc.) | 0.085 | -0.467 |
| gtdb_distance (GTDB phylogeny) | 0.055 | -0.460 |

Two things stand out. First, the 16S phylogeny branch is by far the most necessary, removing it costs almost 5x the R2 that removing either other branch does, and its sufficiency-alone R2 (-0.029) is the closest any single branch gets to the full model's own R2 (0.139), it's carrying most of the real signal. Second, no branch is sufficient alone, every sufficiency number is still negative, meaning the model needs the branches together and isn't just leaning on one of them for its full performance, even though 16S clearly dominates.

## Does the answer differ by regime?

This is where the honest answer gets more qualified. Splitting necessity/sufficiency by doubling-time regime (same full-corpus checkpoint, cross-validated):

**Slow growers (n=111, >=5h doubling time):** the pattern above holds and sharpens further. rrna16s necessity R2 drop is 1.492, the largest necessity effect seen anywhere in this analysis, larger even than the pre-fix number (1.311). gtdb_distance is next (0.296), genomic_traits smallest (0.246). Phylogeny, specifically the 16S branch, dominates for slow growers.

**Fast growers (n=64, <5h doubling time):** the model's absolute fit is much worse here (full R2 = -2.607, worse than the -1.003 for slow growers), and the necessity numbers are noisy and inconsistent, most branches show a negative "drop" (removing them appears to help, a sign of overfitting noise at this per-fold sample size, ~13 test samples per fold, not a real finding). No regime-specific conclusion can honestly be drawn for fast growers from this data. The model simply isn't fitting this regime well enough yet to trust what the ablation says about it.

So the honest answer to "does it differ by regime" is: yes for slow growers, where 16S phylogeny's importance is real, large, and got even larger under the corrected methodology. For fast growers, the data doesn't support a conclusion either way, the model's predictive performance there isn't good enough yet for the necessity/sufficiency breakdown to mean anything.

## Seed robustness check

Everything above was first produced with seed 42, the one used throughout this project. To find out whether that agreement across measurements was real signal or a lucky draw, both checkpoints were retrained from scratch with seed 1 and the entire evaluation suite (benchmark, raw baseline, probing, necessity/sufficiency) was rerun against them, using the identical `--seed 1` for the cross-validation folds too so the comparison isn't just a different random model on the same folds.

**Benchmark (5-fold CV, n=175), R2 (all) / Spearman (all):**

| model | seed 42 | seed 1 |
|---|---|---|
| raw features, no encoder | -0.097 / 0.318 | -0.097 / 0.332 |
| JEPA, labeled-only pretrain | -0.079 / 0.380 | -0.087 / 0.430 |
| JEPA, full-corpus pretrain | -0.055 / 0.448 | -0.092 / 0.367 |

This is the headline that doesn't survive. Under seed 42, full-corpus pretraining beats labeled-only on both metrics. Under seed 1, that ordering reverses: labeled-only beats full-corpus on both R2 and Spearman, and full-corpus is no longer clearly better than the raw-feature baseline at all (R2 -0.092 vs. raw's -0.097, a wash; Spearman 0.367 vs. raw's 0.332, still ahead but by much less than under seed 42). At n=175 with 5-fold CV, roughly 35 species decide each fold's test score, and that's evidently not enough to pin down whether adding the unlabeled GEM corpus to pretraining helps, hurts, or does nothing. The raw-feature baseline itself was the most seed-stable number in the whole comparison, which makes sense, it has no pretraining stage for a different seed to perturb.

What does hold up between seeds:

- **Pretraining beats no pretraining.** Labeled-only pretraining beat the raw-feature baseline on both metrics under both seeds. The encoder is adding value over raw features; that part is real.
- **Probing.** Full-corpus beat labeled-only on both linear and nonlinear probe accuracy and AUC under both seeds (seed 1: linear accuracy 0.774 vs. 0.766, AUC 0.842 vs. 0.803; seed 42: 0.766 vs. 0.718, AUC 0.794 vs. 0.767). This is the one full-corpus-vs-labeled-only comparison that actually replicated.
- **Which branch is most necessary.** In both the whole-corpus and slow-grower regimes, under both seeds, removing rrna16s costs more R2 than removing either other branch. The ranking direction survives even though the necessity/sufficiency R2 magnitudes swing enormously between seeds (see below), which is itself informative: the direction of "16S matters most" looks more trustworthy than any specific number attached to it.

What doesn't hold up:

- **Necessity/sufficiency absolute R2 is wildly seed-sensitive.** The full-corpus checkpoint's whole-corpus R2 was +0.139 under seed 42 and -0.542 under seed 1, a sign flip, not just noise around a number. The slow-grower regime R2 went from -1.003 (seed 42) to -3.373 (seed 1). The fast-grower regime R2 went from -2.607 (seed 42) to -0.106 (seed 1), better under seed 1, actually, which also flips which regime the model fits worse. None of the absolute fit numbers in this analysis should be read as stable; only the branch-importance ranking should be.
- **The "full-corpus pretraining is strictly better" narrative.** It isn't, not reliably, at this labeled-corpus size. See the benchmark table above.

The honest summary: this is a 175-species labeled corpus being evaluated with 5-fold CV, and several of the differences reported as findings earlier in this document are within the noise a single seed change produces. What survives two different seeds, pretraining helps over raw features, probing favors full-corpus pretraining, and the 16S branch is the most necessary one, are the claims worth trusting. What doesn't, whether full-corpus pretraining is worth it over labeled-only for the actual growth-rate benchmark, and any specific necessity/sufficiency R2 value, should be treated as provisional until there's more labeled data to evaluate on.

## What this doesn't yet show

- **R2 is negative almost everywhere.** The model is picking up real rank-order signal (Spearman 0.38-0.45, probing well above chance) but isn't yet a good absolute growth-rate predictor. Both gRodon and Phydon baselines still beat it on Spearman (0.582-0.626 vs. our 0.380-0.448). At 175 species, that gap could close, widen, or reverse with more real labeled data, this isn't a claim that our approach beats the established methods, it doesn't yet.
- **The fast-growth regime is underserved.** 64 species, R2 deeply negative, necessity/sufficiency numbers not trustworthy. Whatever controls fast growth in this model's terms isn't resolvable from the current corpus.
- **The oligotroph/copiotroph probing result, while now real (see research_log.md for the label itself), doesn't by itself explain growth rate.** It shows the joint latent encodes trophic-strategy-relevant structure; it doesn't establish that trophic strategy is what's driving the growth-rate predictions specifically. Those are two different claims and only the first one has real support here.
- **Most of the specific numbers in this document are seed-sensitive at this sample size.** The "Seed robustness check" section above found a sign flip in one headline R2 and a full reversal of the "full-corpus beats labeled-only" benchmark ordering when the run was repeated with a different random seed. Treat any single number here as an estimate with a wide, unquantified error bar, not a precise result, until there's a bigger labeled corpus to narrow it.

## Bottom line

For slow-growing organisms, the 16S phylogeny branch carries more predictive weight than either GTDB phylogeny or genome composition in this model, and that ranking held up not just under cross-validation and a full retraining after a reproducibility bug fix, but under a second random seed too, even though the exact R2 numbers attached to it swing a lot between seeds. That's the most trustworthy result in this document. Pretraining the encoder at all, rather than feeding raw features straight to the growth-rate head, also held up under a second seed. Whether pretraining on the larger unlabeled GEM corpus specifically beats pretraining on the small labeled set alone did not hold up: it looked real under the original seed, and reversed under a second one, so it's reported here as an open question, not a finding, until a bigger labeled corpus can settle it. For fast-growing organisms, no regime-specific claim can be made yet under either seed, the sample is too small and the model's fit too poor and too seed-unstable there. What hasn't changed is that the whole stack still trails gRodon and Phydon on absolute predictive accuracy. The clearest path to a stronger answer is more labeled species (see README's "Known limitations" for exactly where the current 175-species ceiling comes from and what it would take to raise it), not more pipeline engineering or more seeds, the pipeline itself is real, reproducible, and now honestly checked for the kind of seed-luck that a small corpus is prone to.
