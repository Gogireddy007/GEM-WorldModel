# Findings

What this project set out to answer: what controls microbial growth rate, and does the answer differ for fast-growing organisms (doubling time under 5 hours) versus slow-growing ones (5 hours or more). Everything below is from real runs on real data, cross-validated where the sample size demanded it, current as of 2026-08-31. Numbers this small a corpus produces are not the kind you'd put in a paper's abstract, they're honestly reported anyway because that's the actual state of the evidence right now.

## The corpus, briefly

175 species with a real measured growth rate, GTDB tree placement, genomic traits, and 16S data (the true ceiling given how few gRodon/Madin accessions are actual GTDB tree tips, see research_log.md for how that number was arrived at). 51,612 GEM MAGs with real genomic traits and phylogenetic placement, 1,875 of those also with a real 16S sequence, used as an unlabeled pretraining supplement since pretraining doesn't need growth-rate labels.

## Does self-supervised pretraining on the larger unlabeled corpus actually help?

Yes, consistently, across three independent measurements:

1. **Benchmark (cross-validated, n=175):** full-corpus pretraining lifts Spearman rank correlation from 0.380 to 0.425 over labeled-only pretraining. R2 stays roughly flat (-0.078 to -0.057, both negative, this is a small, noisy corpus).
2. **Probing (real trophic label):** linear probe accuracy goes from 0.789 to 0.816, AUC from 0.750 to 0.804. Nonlinear probe goes from 0.711 to 0.842, AUC from 0.747 to 0.827.
3. **Necessity/sufficiency (cross-validated):** full-corpus pretraining gets a positive R2 (0.108) on the whole labeled corpus; labeled-only pretraining doesn't (-0.146).

Three different evaluations, three different metrics, same direction every time. That's about as much confidence as this sample size can support, but it's real and repeated, not a single lucky number.

## What controls growth rate, by branch

Necessity/sufficiency masking (cross-validated, full-corpus checkpoint) on the whole labeled corpus:

| branch | necessity (R2 drop if removed) | sufficiency (R2 alone) |
|---|---|---|
| rrna16s (16S phylogeny) | 0.423 | -0.143 |
| gtdb_distance (GTDB phylogeny) | 0.132 | -0.454 |
| genomic_traits (CUB, GC, genome size, etc.) | 0.120 | -0.573 |

Two things stand out. First, phylogenetic placement (either branch) matters more to the model than genome composition, removing either phylogeny branch hurts more than removing genomic traits. Second, no branch is sufficient alone, every sufficiency number is negative, meaning the model needs the branches together and isn't just leaning on one of them. That's consistent with the architecture actually doing something, not degenerating to a single dominant input.

## Does the answer differ by regime?

This is where the honest answer gets more qualified. Splitting necessity/sufficiency by doubling-time regime (same full-corpus checkpoint, cross-validated):

**Slow growers (n=111, >=5h doubling time):** the pattern above holds and sharpens. rrna16s necessity R2 drop is 1.311, the largest necessity effect seen anywhere in this analysis. gtdb_distance is next (0.602), genomic_traits smallest (0.286). Phylogeny dominates for slow growers, by a wide margin.

**Fast growers (n=64, <5h doubling time):** the model's absolute fit is much worse here (full R2 = -2.538, worse than the -1.149 for slow growers), and the necessity numbers are noisy and inconsistent, some branches show a negative "drop" (removing them appears to help, which is a sign of overfitting noise at this per-fold sample size, ~13 test samples per fold, not a real finding). No regime-specific conclusion can honestly be drawn for fast growers from this data. The model simply isn't fitting this regime well enough yet to trust what the ablation says about it.

So the honest answer to "does it differ by regime" is: yes for slow growers, where phylogeny's importance is real and gets stronger than the pooled result suggests. For fast growers, the data doesn't support a conclusion either way, the model's predictive performance there isn't good enough yet for the necessity/sufficiency breakdown to mean anything.

## What this doesn't yet show

- **R2 is negative almost everywhere.** The model is picking up real rank-order signal (Spearman 0.38-0.47, probing well above chance) but isn't yet a good absolute growth-rate predictor. Both gRodon and Phydon baselines still beat it on Spearman (0.582-0.626 vs. our 0.380-0.425). At 175 species, that gap could close, widen, or reverse with more real labeled data, this isn't a claim that our approach beats the established methods, it doesn't yet.
- **The fast-growth regime is underserved.** 64 species, R2 deeply negative, necessity/sufficiency numbers not trustworthy. Whatever controls fast growth in this model's terms isn't resolvable from the current corpus.
- **The oligotroph/copiotroph probing result, while now real (see research_log.md for the label itself), doesn't by itself explain growth rate.** It shows the joint latent encodes trophic-strategy-relevant structure; it doesn't establish that trophic strategy is what's driving the growth-rate predictions specifically. Those are two different claims and only the first one has real support here.

## Bottom line

For slow-growing organisms, phylogenetic placement carries more real predictive weight than genome composition in this model, and that finding held up under cross-validation and got stronger, not weaker, when isolated to that regime. For fast-growing organisms, no honest regime-specific claim can be made yet, the sample is too small and the model's fit too poor there. Combining self-supervised pretraining on the much larger unlabeled GEM corpus with the small labeled set helps, consistently, across three independent measurements, though it hasn't yet closed the gap with gRodon or Phydon on absolute predictive accuracy. The clearest path to a stronger answer is more labeled species (see README's "Known limitations" for exactly where the current 175-species ceiling comes from and what it would take to raise it), not more pipeline engineering, the pipeline itself is real and working end to end.
