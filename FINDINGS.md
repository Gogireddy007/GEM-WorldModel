# Findings

What this project set out to answer: what controls microbial growth rate, and does the answer differ for fast-growing organisms (doubling time under 5 hours) versus slow-growing ones (5 hours or more). Everything below is from real runs on real data, cross-validated where the sample size demanded it, current as of 2026-09-02. Numbers this small a corpus produces are not the kind you'd put in a paper's abstract, they're honestly reported anyway because that's the actual state of the evidence right now.

A real bug was found and fixed right before this version: nothing in the codebase ever called `.eval()` on the model, so its dropout layers were active during every inference call, including the target encoder during pretraining itself, where it should never have been active at all (see research_log.md, 2026-09-02). Every checkpoint and every number below was regenerated after the fix. The core finding held up, and in a couple of places got stronger, not weaker.

A follow-up question then had to be asked directly: were those numbers real, or just a lucky seed? Every checkpoint and every evaluation below was rerun end to end across seven random seeds total (42, 1, 7, 13, 99, 123, 2024) to check. The answer is mixed, and it's reported honestly, with exact replication fractions, in "Seed robustness check" further down: some of what's claimed here holds up in the large majority of seeds, one of the headline comparisons is close to a coin flip.

## The corpus, briefly

175 species with a real measured growth rate, GTDB tree placement, genomic traits, and 16S data (the true ceiling given how few gRodon/Madin accessions are actual GTDB tree tips, see research_log.md for how that number was arrived at). 51,612 GEM MAGs with real genomic traits and phylogenetic placement, 1,875 of those also with a real 16S sequence, used as an unlabeled pretraining supplement since pretraining doesn't need growth-rate labels.

## Does self-supervised pretraining on the larger unlabeled corpus actually help?

Under the seed this project has used throughout (seed 42), yes, consistently, across three independent measurements:

1. **Benchmark (5-fold cross-validated, n=175):** full-corpus pretraining lifts Spearman rank correlation from 0.380 to 0.448 over labeled-only pretraining. R2 stays roughly flat and negative (-0.079 to -0.055, this is a small, noisy corpus).
2. **Probing (real trophic label, 5-fold cross-validated, n=124):** linear probe accuracy goes from 0.718 to 0.766, AUC from 0.767 to 0.794. Nonlinear probe goes from 0.734 to 0.750, AUC from 0.801 to 0.794 (essentially flat there, the one place this measurement didn't clearly favor full-corpus).
3. **Necessity/sufficiency (5-fold cross-validated):** full-corpus pretraining gets a real positive R2 (0.139) on the whole labeled corpus; labeled-only pretraining doesn't (-0.164).

Three different evaluations, three different metrics, the same direction in five of six numbers under this one seed. But six more seeds were run specifically to check whether that agreement was real or a coincidence of this particular random draw, and it wasn't fully real: see "Seed robustness check" below. Across all 7 seeds, the benchmark comparison (item 1, full-corpus vs. labeled-only) holds in only 3-4 of 7 seeds depending on the metric, a coin flip, not a finding. The probing comparison (item 2) held up in the two seeds it was directly compared on. The qualitative necessity/sufficiency ranking (item 3, which branch matters most) held in 6 of 7 seeds.

## Is the JEPA encoder itself earning its place?

This has to be checked directly, not assumed. Ran the raw (standardized) branch features straight into a growth-rate head, no encoder, no pretraining at all, through the exact same k-fold cross-validation (identical fold membership) as everything else:

| model | R2 (all) | Spearman (all) |
|---|---|---|
| raw features, no encoder | -0.097 | 0.318 |
| JEPA, labeled-only pretrain | -0.079 | 0.380 |
| JEPA, full-corpus pretrain | -0.055 | 0.448 |
| gRodon reproduction | -0.032 | 0.582 |
| Phydon reproduction | 0.034 | 0.626 |

Both metrics improve monotonically going from raw features to labeled-only pretraining to full-corpus pretraining, under seed 42. Across all 7 seeds tested, the first step holds up well: labeled-only pretraining beats the raw baseline's R2 in 7/7 seeds and its Spearman in 6/7. The second step, full-corpus beating labeled-only, does not hold up: see "Seed robustness check" below. So the honest claim is narrower than it first looked: pretraining the encoder at all is earning its place, reliably; more pretraining data (the full unlabeled GEM corpus vs. the labeled set alone) is not reliably earning more, at least not yet at this labeled-corpus size.

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

Everything above was first produced with seed 42, the one used throughout this project. To find out whether that agreement across measurements was real signal or a lucky draw, both checkpoints were retrained from scratch and the entire evaluation suite (benchmark, raw baseline, necessity/sufficiency, and for the first two additional seeds, probing) was rerun against six more seeds: 1, 7, 13, 99, 123, 2024. Each seed used the identical `--seed N` for the cross-validation folds too, so the comparison isn't a different random model evaluated on the same folds, it's a genuinely independent draw each time. Seven seeds, not two, so the fractions below are real replication rates, not a single up-or-down check.

**Benchmark (5-fold CV, n=175), R2 (all) / Spearman (all), by seed:**

| seed | raw, no encoder | JEPA, labeled-only | JEPA, full-corpus |
|---|---|---|---|
| 42 | -0.097 / 0.318 | -0.079 / 0.380 | -0.055 / 0.448 |
| 1 | -0.097 / 0.332 | -0.087 / 0.430 | -0.092 / 0.367 |
| 7 | -0.097 / 0.188 | -0.076 / 0.351 | -0.058 / 0.390 |
| 13 | -0.097 / 0.216 | -0.085 / 0.309 | -0.088 / 0.293 |
| 99 | -0.097 / 0.190 | -0.091 / 0.211 | -0.091 / 0.110 |
| 123 | -0.097 / 0.296 | -0.092 / -0.322 | -0.071 / 0.294 |
| 2024 | -0.097 / 0.023 | -0.089 / 0.404 | -0.083 / 0.285 |

**Replication rate per claim, across all 7 seeds:**

| claim | how many seeds it held in |
|---|---|
| full-corpus pretraining beats labeled-only (R2) | 4 / 7 (42, 7, 123, 2024 win; 1, 13, 99 lose) |
| full-corpus pretraining beats labeled-only (Spearman) | 3 / 7 (42, 7, 123 win; 1, 13, 99, 2024 lose) |
| labeled-only pretraining beats raw baseline (R2) | 7 / 7 |
| labeled-only pretraining beats raw baseline (Spearman) | 6 / 7 (loses only at seed 123) |
| full-corpus pretraining beats raw baseline (R2) | 7 / 7 (margins tiny in some seeds, e.g. seed 1: -0.092 vs -0.097) |
| full-corpus pretraining beats raw baseline (Spearman) | 5 / 7 (loses at seeds 99 and 123) |
| rrna16s is the single most necessary branch for slow growers | 6 / 7 (loses only at seed 99, where gtdb_distance edges it 1.250 vs 1.122, a near-tie, not a reversal) |
| necessity/sufficiency whole-corpus R2 is positive (full-corpus checkpoint) | 2 / 7 (only seeds 42 and 7; the other five are all negative, one as low as -0.542) |

This is now a genuinely settled picture, not a single-alternate-seed guess. "Full-corpus beats labeled-only" is a real coin flip, not a finding, on either metric. "Pretraining beats raw features" is close to universal on R2 and strong on Spearman, that's real. "16S is the most necessary branch for slow growers" replicates in 6 of 7 seeds and the one exception is a near-tie, not a contradiction, that's the strongest claim in this document. The specific necessity/sufficiency R2 numbers are not stable at all, sign-flipping in 5 of 7 seeds, so no absolute number from that analysis should be quoted without this caveat attached.

What does hold up across (most or all) seeds:

- **Pretraining beats no pretraining.** Both labeled-only and full-corpus pretraining beat the raw-feature baseline's R2 in every single seed tested, and beat its Spearman in the large majority. The encoder is adding value over raw features; that part is real and now checked seven independent ways.
- **Which branch is most necessary.** In the slow-grower regime, under 6 of 7 seeds, removing rrna16s costs more R2 than removing either other branch, and the one exception is a near-tie. The ranking direction survives even though the necessity/sufficiency R2 magnitudes swing enormously between seeds (see below), which is itself informative: the direction of "16S matters most" is far more trustworthy than any specific number attached to it.

What doesn't hold up:

- **Necessity/sufficiency absolute R2 is wildly seed-sensitive.** The full-corpus checkpoint's whole-corpus R2 ranged from +0.139 (seed 42) to -0.542 (seed 1) across the 7 seeds, a sign flip in 5 of 7 runs, not just noise around a number. None of the absolute fit numbers in this analysis should be read as stable; only the branch-importance ranking should be.
- **The "full-corpus pretraining is strictly better" narrative.** It isn't. 4/7 on R2, 3/7 on Spearman is not a real effect at this labeled-corpus size, it's noise that happened to point the same direction the first two times it was checked (seeds 42 and 1 were the first comparison made, and even that pair alone was already a reversal before the other five seeds confirmed it wasn't a fluke in the other direction either).

The honest summary: this is a 175-species labeled corpus being evaluated with 5-fold CV, and several of the differences originally reported as findings are within the noise a seed change produces, now confirmed with seven independent draws rather than assumed from one or guessed from two. What survives, pretraining helps over raw features and the 16S branch is the most necessary one, are the claims worth trusting. What doesn't, whether full-corpus pretraining is worth it over labeled-only for the actual growth-rate benchmark, and any specific necessity/sufficiency R2 value, should be treated as unresolved until there's more labeled data to evaluate on, not as a coin flip that more seeds will eventually resolve in one direction, there's no reason to expect it will, the sample size itself is the limit.

## Does a larger labeled corpus change anything?

Everything above tops out at 175 species, the true ceiling of exact GTDB-tree-tip placement for the gRodon/Madin growth-rate corpus. A second way to grow it was built and run: 129 more species that have real growth-rate data and real GTDB taxonomy but aren't themselves a tree tip, given an *approximate* phylogenetic embedding instead, the centroid of the embeddings of tree-tip species sharing the same GTDB genus (`features/phylogeny.py:genus_centroid_embeddings`). Every other feature (genome size, GC, CUB, 16S) for these 129 is real, freshly pulled from NCBI, not approximated. This produces a 304-species corpus (175 exact-tip + 129 genus-centroid-approximate), and the full pipeline was rerun against it at seed 42 for a direct before/after comparison.

**Benchmark (5-fold CV), R2 (all) / Spearman (all), n=175 vs. n=304:**

| model | n=175 | n=304 |
|---|---|---|
| raw features, no encoder | -0.097 / 0.318 | -0.090 / 0.149 |
| JEPA, labeled-only pretrain | -0.079 / 0.380 | -0.076 / 0.306 |
| JEPA, full-corpus pretrain | -0.055 / 0.448 | -0.044 / 0.405 |
| gRodon reproduction | -0.032 / 0.582 | -0.032 / 0.613 |
| Phydon reproduction | 0.034 / 0.626 | 0.100 / 0.775 |

R2 improved slightly across every model with the larger corpus. Spearman did not: it dropped for all three of our variants (raw, labeled-only, full-corpus) while jumping substantially for both baselines, especially Phydon (0.626 to 0.775, and its R2 turned positive for the first time anywhere in this project). The 129 new species skew toward well-studied lab and industrial organisms (E. coli relatives, Pseudomonas, Klebsiella, Vibrio, and similar), exactly the kind of genome gRodon/Phydon's own published training and validation likely covers well. That's a plausible, honest explanation for why the established tools got sharply better on this expanded set while this project's model's rank-order performance got worse, not evidence the larger corpus is bad, but a real sign that adding species this way changed corpus composition, not just corpus size, and the two effects aren't separable from this one run. The gap between this project's model and the established baselines widened on the expanded corpus, it did not close.

Full-corpus pretraining beat labeled-only pretraining on both metrics at n=304 (R2 -0.044 vs. -0.076, Spearman 0.405 vs. 0.306), the same direction as the original seed-42 result on n=175. That single comparison doesn't by itself establish this claim is more robust now, the 7-seed check on n=175 already showed this specific comparison is a coin flip across seeds; a proper answer would need the same multi-seed check repeated on n=304, which hasn't been done. What can be said now is only: at seed 42, on the larger corpus, the direction is unchanged.

**Necessity/sufficiency (full-corpus checkpoint), n=304:** whole-corpus R2 is positive again, 0.121, similar order of magnitude to the original n=175 result (0.139). In the slow-grower regime (n=174 now, up from 111), rrna16s is again by far the most necessary branch, R2 drop 1.082 versus 0.197 (genomic_traits) and 0.107 (gtdb_distance), an even wider margin than at n=175. This is a real corroboration of the project's strongest finding on a corpus nearly double the size, using a mix of exact and approximate phylogenetic placements, and it held.

**Probing (real trophic label, n=226, up from 124):** linear probe accuracy 0.761 (labeled-only) / 0.819 (full-corpus), both up from n=175's 0.718/0.766. Nonlinear probe went 0.810/0.796, labeled-only now edges out full-corpus, a reversal from n=175 where full-corpus led narrowly on both probe types. Mixed, not a clean confirmation either way.

Net read: the encoder-beats-raw-features finding and the 16S-branch-dominance finding both hold up, and in the 16S case, the margin actually widened. The full-corpus-beats-labeled-only comparison pointed the same direction again but wasn't re-checked for seed stability at this corpus size, so it isn't more trustworthy than before, just not contradicted. The genuinely new and slightly uncomfortable result is the Spearman regression alongside gRodon/Phydon's sharp improvement, a reminder that "more labeled data" via genus-centroid approximation isn't a free win, it changes what's in the corpus, and that composition shift mattered here as much as the extra count did.

## What this doesn't yet show

- **R2 is negative almost everywhere.** The model is picking up real rank-order signal (Spearman 0.38-0.45, probing well above chance) but isn't yet a good absolute growth-rate predictor. Both gRodon and Phydon baselines still beat it on Spearman (0.582-0.626 vs. our 0.380-0.448). At 175 species, that gap could close, widen, or reverse with more real labeled data, this isn't a claim that our approach beats the established methods, it doesn't yet.
- **The fast-growth regime is underserved.** 64 species, R2 deeply negative, necessity/sufficiency numbers not trustworthy. Whatever controls fast growth in this model's terms isn't resolvable from the current corpus.
- **The oligotroph/copiotroph probing result, while now real (see research_log.md for the label itself), doesn't by itself explain growth rate.** It shows the joint latent encodes trophic-strategy-relevant structure; it doesn't establish that trophic strategy is what's driving the growth-rate predictions specifically. Those are two different claims and only the first one has real support here.
- **Most of the specific numbers in this document are seed-sensitive at this sample size.** The "Seed robustness check" section above, now run across 7 independent seeds rather than one or two, found a sign flip in the headline necessity/sufficiency R2 in 5 of 7 runs, and the "full-corpus beats labeled-only" benchmark ordering holding in only 3-4 of 7 seeds depending on the metric, a coin flip. Treat any single number here as an estimate with a wide, unquantified error bar, not a precise result, until there's a bigger labeled corpus to narrow it.

## Bottom line

For slow-growing organisms, the 16S phylogeny branch carries more predictive weight than either GTDB phylogeny or genome composition in this model, and that ranking held up not just under cross-validation and a full retraining after a reproducibility bug fix, but in 6 of 7 independent random seeds, with the one exception a near-tie rather than a reversal, even though the exact R2 numbers attached to it swing a lot between seeds. That's the most trustworthy result in this document. Pretraining the encoder at all, rather than feeding raw features straight to the growth-rate head, also held up in essentially every seed tested (7/7 on R2). Whether pretraining on the larger unlabeled GEM corpus specifically beats pretraining on the small labeled set alone did not hold up: it won in 4 of 7 seeds on R2 and 3 of 7 on Spearman, indistinguishable from chance, so it's reported here as an open question, not a finding, and there's no reason to expect more seeds would resolve it one way, the 175-species sample size is the actual limit, not bad luck. For fast-growing organisms, no regime-specific claim can be made yet under any seed tested, the sample is too small and the model's fit too poor and too seed-unstable there. What hasn't changed is that the whole stack still trails gRodon and Phydon on absolute predictive accuracy. The clearest path to a stronger answer is more labeled species (see README's "Known limitations" for exactly where the current 175-species ceiling comes from and what it would take to raise it), not more pipeline engineering or more seeds, the pipeline itself is real, reproducible, and now honestly checked across seven independent random draws for the kind of seed-luck that a small corpus is prone to.
