# Research Log

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
