# Story

## Working title

**How fast do uncultured microbes grow? Predicting maximal growth rates for metagenome-assembled genomes, with lineage-held-out evaluation and an audit of the training data**

Shorter alternatives:
- Predicting growth rates of uncultured microbes from their genomes, and how far the predictions can be trusted
- Growth rate prediction for metagenome-assembled genomes under honest evaluation

## The paper in one paragraph

Most microbes are known only from genomes assembled out of environmental DNA, and how fast they can grow is one of the
first things one would want to know about them. Existing predictors (gRodon, Phydon) are evaluated mostly on species that
have close relatives in the training data and are given the measured growth temperature, which uncultured genomes do not
have. We built a predictor that uses only the genome: codon usage statistics, genes for energy metabolism and lifestyle,
placement on the GTDB tree, and the 16S gene. We measure its accuracy with whole species, genera, families or orders held
out, which is the situation for most environmental genomes, and report an error and an 80% range for each prediction by
how novel the genome is. On the way we found that the training compilation contains values that are not maximal
growth rates: ten of them, from five sources, and four from a single study were enough to create an apparent biological
effect that vanished once they were removed. With the
same inputs, our model is more accurate than the released gRodon and Phydon. We apply it to the 9,139 high-quality genomes of
the Genomes from Earth's Microbiomes catalog, and use it to test whether slow growers are slow because of how they make
energy: once relatedness is controlled, they are not, at least among cultured organisms; what separates them is a lack of
the machinery for exploiting rich, changing conditions.

## Contributions (each must survive the claim audit)

1. **Evaluation that matches how the predictions are used.** Accuracy measured with species, genera, families and orders
   held out, nested tuning so the headline numbers are not optimistic, and a per-genome error and range by novelty tier.
2. **A training-data audit.** Ten labels in the Madin compilation that are not maximal growth rates, found by tracing
   values to their source records and by a study-level check; a positive control shows the check finds them. One source
   alone produced a fivefold "effect" of methane oxidation on growth.
3. **A fair comparison with the released tools.** gRodon and Phydon run as released, on the same genes and held-out
   species, with Phydon's training data stripped of each test fold. Without the measured temperature our model has a lower
   error than either tool given it; with equal inputs it is better on error, ranking and slow-growth AUC.
4. **A genome-only predictor that holds up on real MAGs.** The error from MAGs is close to the error from isolate genomes
   of the same species.
5. **A controlled test of the low-energy explanation of slow growth**, with phylogenetic regression and robustness
   checks: energy metabolism genes do not explain growth rate; slow growers have fewer transporter genes (the regulator
   association depends on the Pseudomonadota and is not claimed).
6. **A resource**: predicted doubling time, growth temperature, error and range for 9,139 GEM genomes.

## What reviewers will push on, and the answer

| Likely objection | Answer in the paper |
|---|---|
| "Accuracy is inflated by relatives in the training set" | Whole lineages held out at four levels; nested tuning; per-tier error reported and used for each prediction |
| "You beat reimplementations, not the real tools" | Released gRodon and Phydon, our genes, our folds, Phydon's training data stripped per fold; we state their remaining advantages |
| "The comparison is unfair to gRodon and Phydon" | We show both settings: without the measured temperature and with it; our model is ahead in both on error |
| "Labels are noisy" | Label audit with a positive control; exclusions listed with evidence; sensitivity of results to them |
| "Predictions on MAGs are not validated" | 717 MAGs from 47 species with measured rates, species held out, compared with isolate genomes |
| "The model just learns phylogeny" | Accuracy for new orders; ablations of each feature block; the phylogenetic regression for the biological claim |
| "Slow growth is underpredicted" | Stated plainly and quantified: slow growers are ranked well (AUC) but rarely predicted above a day for new lineages |
| "Why not deep learning?" | 754 labeled genomes; a JEPA encoder was tried earlier and lost to tree models (claim 20, decide in phase 5 whether to report) |

## Figures (main text, about five)

1. Overview: data sources, label audit, features, lineage-held-out evaluation, GEM application (schematic)
2. Accuracy by novelty tier for the three targets, with nested estimates and per-prediction ranges
3. Comparison with released gRodon and Phydon, with and without the measured temperature
4. Real MAGs against isolates of the same species; GEM predictions by habitat and lineage
5. The low-energy test: effects before and after controlling for relatedness

## Tables

1. Training data after the audit: sources, counts, exclusions
2. Ablations (gRodon statistics, learner, added species, energy features, predicted temperature), nested, orders held out

## Supplement

Label exclusions with evidence; literature strains; feature definitions with Pfam accessions and the checks against known
organisms; full accuracy tables; all phylogenetic test results and robustness checks; negative results (syntrophy marker,
energy features, predicted temperature, earlier JEPA encoder); GEM prediction table.
