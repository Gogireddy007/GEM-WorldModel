# Findings

What this project set out to answer: what controls microbial growth rate, and does the answer differ for fast-growing organisms (doubling time under 5 hours) versus slow-growing ones (5 hours or more). Everything below is from real runs on real data, cross-validated where the sample size demanded it, current as of 2026-09-02. Numbers this small a corpus produces are not the kind you'd put in a paper's abstract, they're honestly reported anyway because that's the actual state of the evidence right now.

**Update, 2026-09-17, version 1.0 and after:** a stronger baseline changed the answer to one of this document's central questions. See "A stronger head changes the picture" below before reading the JEPA-versus-raw-features sections that follow as the final word, they were true for the setup they were tested under, but that setup used a weak growth-rate head on both sides of the comparison. With a proper head, the ordering flips.

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

## A stronger head changes the picture

Everything above compares models that all share one thing: a small neural network head, trained by gradient descent, sitting on top of whatever features feed it. That head was never checked against a real alternative until version 1.0. It should have been.

Two things were tested together, on the 304-species corpus, same folds, same seven seeds already used everywhere in this document: swapping the small fine-tuned head for gradient-boosted trees, and checking whether the pretrained JEPA encoder still added anything once that swap was made.

**Encoder plus gradient-boosted trees, versus the old encoder plus small fine-tuned head, at seed 42:** R2 went from -0.044 to 0.096, Spearman from 0.405 to 0.687. Checked across all seven seeds, R2 came out positive in six of seven, and Spearman held in a tight 0.60 to 0.69 band every single time, far steadier than the old head ever managed.

That result did not survive its own control. **Raw features, no encoder at all, plus gradient-boosted trees, beat the encoder version in every one of the seven seeds**, on both R2 and Spearman. The gain was never coming from the encoder. It was the head all along.

Run the full way, with the fast/slow regime breakdown and against both established tools, raw features plus gradient-boosted trees is the best result this project has produced:

| model | R2 (all, seed 42) | Spearman (all, seed 42) | R2 beats gRodon | R2 beats Phydon |
|---|---|---|---|---|
| raw features, gradient-boosted trees | 0.132 | 0.760 | 7 / 7 seeds | 7 / 7 seeds |
| gRodon reproduction | -0.032 | 0.613 | - | - |
| Phydon reproduction | 0.100 | 0.775 | - | - |

This is the first result anywhere in this project that beats gRodon outright, on both metrics, in every seed tested, and beats Phydon on R2 in every seed too, though Phydon still edges it slightly on Spearman more often than not (wins 5 of 7). Nothing about the JEPA architecture, the pretraining, or the branch masking was needed to get here, standardized raw genome features and a tree-based model got further than all of it.

Two things this does not fix on its own, both followed up the same day. The fast-growth regime is still very poor, R2 around -26 in that regime, unchanged from every earlier version of this model. Looking at the actual mispredictions: 72% of fast growers get predicted slower than they really are, and it's a real, traceable cause, not noise. The whole corpus's median doubling time is 7 hours, but the fast regime's true median is 1.71 hours, and a model trained to minimize squared error over that whole skewed range naturally gets pulled toward the middle and compresses fast growers upward. This is a property of the training loss and the label distribution, not a mistake in the features, head, or encoder, and it needs a loss change or reweighting to fix, not more tuning of what's already here.

The tree model's own feature importance (roughly 45% GTDB phylogeny, 28% each for 16S and genome traits across the whole corpus at seed 42) doesn't fully agree with the necessity/sufficiency finding elsewhere in this document that 16S carried the most weight for slow growers specifically. Splitting the importance calculation by regime instead of averaging over the whole corpus narrows the gap: within slow growers alone, rrna16s importance rises to 0.41, close behind gtdb_distance at 0.47, the same direction as the necessity/sufficiency result, just not the same ranking. Real, partial agreement between the two methods, not a full match, both numbers are kept here rather than picking the one that sounds more consistent.

The honest updated recommendation: for anyone who wants the most accurate prediction this project can currently produce, use raw features with gradient-boosted trees, not the JEPA pipeline. The JEPA architecture and everything built around it remains useful for the interpretability questions, what matters and by how much, but as a predictor on its own it has been outperformed by something far simpler.

## Does blending with gRodon and Phydon help further?

Checked next, the same way, across all seven seeds, with blend weights fit honestly (never using a fold's own labels to pick the weights that score it, see research_log.md for exactly how). The answer is a real trade-off, not a clean win:

| | overall R2 | overall Spearman | worst fast-regime R2 across all seven seeds |
|---|---|---|---|
| gradient-boosted trees alone | best in 7 / 7 seeds | never best | -109 |
| Phydon alone | worst | mixed | -627 |
| blend of all three | middle, every seed | best in 7 / 7 seeds | -28.6 |

The blend never beats plain gradient-boosted trees on overall R2. But it wins overall Spearman in every seed tested, beating both individual models each time, and it substantially tempers the fast-growth regime's failure, the single worst and most persistent problem this project has had. Plain gradient-boosted trees and Phydon both have seeds where the fast regime's R2 falls into the hundreds of negative points, the blend's worst case across all seven seeds is a comparatively tame -28.6. Still a real weakness, not solved, but meaningfully less catastrophic every single time.

Which one to use depends on the goal. For the single best overall R2, use gradient-boosted trees alone. For a model that ranks organisms more reliably and doesn't fall apart as badly on fast growers, the blend is the better choice. Both are kept here rather than declaring one the winner, because "better" depends on what the number is actually going to be used for.

## Where exactly is the training data thin, and does patching it help?

A finer scan of the 304-species corpus (eight bins each way instead of six) found ten regions where the labeled species are dominated by one genus or one narrow growth condition, not just the two spotted earlier by eye. Checked which of those ten actually matter by counting how many real GEM catalog genomes fall into each one. Two stood out:

- The known high GC, small genome, thermophile-heavy region: 1,580 real genomes, 3.0% of the entire 52,515-genome catalog.
- A medium-high GC, 4-5.7 million base pair region that's 50% Vibrio in the training set: 918 real genomes, 1.7% of the catalog, not identified before this check.

Vibrio species are some of the fastest-growing bacteria known, the opposite concern from the thermophile region. Checked it directly against real predictions from the high-quality genome set: genomes landing in this region get a median predicted doubling time of 0.46 hours, against 0.60 hours for everything else. Real and measurable, in the expected direction.

Went looking for a fix. Found 92 real, labeled species with a published growth rate that had never been added to the corpus, only because they lacked a genus-level tree match, the requirement the earlier expansion used. That requirement turned out to be stricter than necessary, only the phylogeny branch needs a tree placement at all. Built a family-level fallback for that branch and pulled real NCBI data for the 39 of those 92 species that were addable even with family-level matching (the other 53 have no match at genus or family level anywhere in the tree).

Checked directly whether any of the 39 new species land in either known gap region before running anything further: none do, zero for zero. Ran the benchmark anyway, checked across four seeds, and the 343-species corpus came out flat or worse than the 304-species one in every seed, R2 worse in two of four, Spearman worse in all four, never better in any. A real, checked, negative result: adding real, legitimately labeled species that don't happen to land where the actual problem is does not fix the problem, and adds enough noise to make things slightly worse on average.

The honest state of this at the time it was written: the two gaps are real, quantified, and named, but nothing reachable seemed to fill them. That turned out to be one search short of true, see below.

### A real search found gap-filling species, and even they did not clearly help

Checked three outside sources for real growth-rate data beyond gRodon and Madin. BacDive, the largest standardized bacterial phenotype database that exists, does not have a doubling-time field at all, checked directly against its own documentation. EGGO's 217,074 growth-rate values are gRodon's own predictions, not real measurements, using them would mean training this project's model to imitate gRodon rather than learn from data. A recent bioRxiv preprint with a promising-looking dataset could not be verified, the site blocked every access attempt.

The real find came from a source already partly in this project: Madin et al. 2020's full published dataset (14,893 species, already cached locally for a different earlier use), which has 502 species with a real doubling time, genome size, and GC content already computed, far more than the roughly 389 species this project's existing pipeline pulls from a narrower subset. Checking those 502 directly against the two known gap regions found 11 real species landing exactly there, with genuinely mixed fast and slow growth rates, not more of the bias already in those regions. Ten resolved to real NCBI accessions, and all ten turned out to already be exact GTDB tree tips once matched against GTDB's own pinned representative genome versions, not an approximation at all. Pulled real data for all ten and rebuilt the corpus, 353 species total.

Benchmarked the same way as everything else, four seeds, against the 304 and 343-species corpora:

| seed | 304 R2 | 343 R2 | 353 (gap-filled) R2 | 304 Spearman | 343 Spearman | 353 Spearman |
|---|---|---|---|---|---|---|
| 42 | 0.132 | 0.135 | 0.094 | 0.760 | 0.740 | 0.668 |
| 1 | 0.086 | 0.086 | 0.101 | 0.733 | 0.720 | 0.677 |
| 7 | 0.126 | 0.077 | 0.130 | 0.760 | 0.696 | 0.686 |
| 13 | 0.189 | 0.100 | 0.064 | 0.771 | 0.734 | 0.673 |

Spearman is worse in all four seeds against both comparisons. R2 is mixed, never a clean win. Even a carefully targeted, real, well-motivated addition did not clearly help. The likely reason: ten species is a small addition, about 3% more data, and they were deliberately chosen to look different from their neighbors, which is exactly what makes a species hard to predict correctly when it lands in a held-out cross-validation fold with nothing like it in the training folds. That is the same small-sample-size ceiling this project runs into everywhere else, just showing up in a new place.

Following through on the standard this check was held to, use it only if it helps: it did not clearly help, so `features_sample_gapfilled.csv` is not adopted as the corpus default. It stays available and documented, a real, honestly negative result from a real, well-targeted attempt, not hidden because the outcome wasn't the hoped-for one.

## Is requiring an exact tree tip even the right rule anymore?

This project has required an exact GTDB tree-tip placement as the default ever since the original 175-species corpus. With real evidence now in hand about how approximate placement performs, that assumption was checked directly rather than left standing: the same benchmark, same four seeds, run on the 175-species exact-only corpus, the 304-species corpus with genus-level approximation, and the 343-species corpus with family-level approximation added on top, side by side.

| seed | 175, exact tip only | 304, + genus approximation | 343, + family approximation |
|---|---|---|---|
| 42 | 0.083 | 0.132 | 0.135 |
| 1 | 0.040 | 0.086 | 0.086 |
| 7 | 0.067 | 0.126 | 0.077 |
| 13 | 0.154 | 0.189 | 0.100 |

(R2, raw features plus gradient-boosted trees, the current best model)

Genus-level approximation is a clean win, R2 improves in all four seeds with no exceptions, and Spearman improves in all four too. This reverses an earlier finding in this document: under the old, fine-tuned JEPA pipeline, this same expansion looked like it hurt Spearman. Under the model that actually matters now, it clearly helps. Family-level approximation is a net negative in three of the four seeds, confirming genus and family level placement are not equally trustworthy, family level is not just a weaker approximation, in this test it was worse than not adding those species at all.

The updated, evidence-based answer: exact-tip-only should not be the default going forward. Genus-level relaxation should be the standard, it is a real, repeatedly confirmed improvement on the model that matters. Family-level relaxation should not be used by default without a stronger reason to trust a specific batch of family-matched species than availability alone.

## What this doesn't yet show

- **R2 is negative almost everywhere.** The model is picking up real rank-order signal (Spearman 0.38-0.45, probing well above chance) but isn't yet a good absolute growth-rate predictor. Both gRodon and Phydon baselines still beat it on Spearman (0.582-0.626 vs. our 0.380-0.448). At 175 species, that gap could close, widen, or reverse with more real labeled data, this isn't a claim that our approach beats the established methods, it doesn't yet.
- **The fast-growth regime is underserved.** 64 species, R2 deeply negative, necessity/sufficiency numbers not trustworthy. Whatever controls fast growth in this model's terms isn't resolvable from the current corpus.
- **The oligotroph/copiotroph probing result, while now real (see research_log.md for the label itself), doesn't by itself explain growth rate.** It shows the joint latent encodes trophic-strategy-relevant structure; it doesn't establish that trophic strategy is what's driving the growth-rate predictions specifically. Those are two different claims and only the first one has real support here.
- **Most of the specific numbers in this document are seed-sensitive at this sample size.** The "Seed robustness check" section above, now run across 7 independent seeds rather than one or two, found a sign flip in the headline necessity/sufficiency R2 in 5 of 7 runs, and the "full-corpus beats labeled-only" benchmark ordering holding in only 3-4 of 7 seeds depending on the metric, a coin flip. Treat any single number here as an estimate with a wide, unquantified error bar, not a precise result, until there's a bigger labeled corpus to narrow it.

## Bottom line

**Update, version 1.0:** the paragraph below was true for the JEPA pipeline as it stood through the seed robustness check. It is no longer the most accurate thing this project can produce, see "A stronger head changes the picture" above. Raw features plus gradient-boosted trees now beats gRodon outright and matches or beats Phydon, something nothing in this paragraph could yet claim. The paragraph is kept as written because it was an honest account of that stage of the work, not because it is still the final answer.

For slow-growing organisms, the 16S phylogeny branch carries more predictive weight than either GTDB phylogeny or genome composition in this model, and that ranking held up not just under cross-validation and a full retraining after a reproducibility bug fix, but in 6 of 7 independent random seeds, with the one exception a near-tie rather than a reversal, even though the exact R2 numbers attached to it swing a lot between seeds. That's the most trustworthy result in this document. Pretraining the encoder at all, rather than feeding raw features straight to the growth-rate head, also held up in essentially every seed tested (7/7 on R2). Whether pretraining on the larger unlabeled GEM corpus specifically beats pretraining on the small labeled set alone did not hold up: it won in 4 of 7 seeds on R2 and 3 of 7 on Spearman, indistinguishable from chance, so it's reported here as an open question, not a finding, and there's no reason to expect more seeds would resolve it one way, the 175-species sample size is the actual limit, not bad luck. For fast-growing organisms, no regime-specific claim can be made yet under any seed tested, the sample is too small and the model's fit too poor and too seed-unstable there. What hasn't changed is that the whole stack still trails gRodon and Phydon on absolute predictive accuracy. The clearest path to a stronger answer is more labeled species (see README's "Known limitations" for exactly where the current 175-species ceiling comes from and what it would take to raise it), not more pipeline engineering or more seeds, the pipeline itself is real, reproducible, and now honestly checked across seven independent random draws for the kind of seed-luck that a small corpus is prone to.
