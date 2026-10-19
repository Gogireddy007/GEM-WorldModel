# GEM-WorldModel

This project tries to predict how fast a microbe grows just from its genome, no lab measurement needed.

The idea: take a genome, pull out three things from it (basic traits like GC content and codon usage, where it
sits on the tree of life, and its 16S gene), train a model on real species that already have a measured growth
rate, and use that model to predict growth rate for genomes that don't. The real research question behind it is
what actually controls microbial growth rate, and whether the answer is different for fast growers versus slow
ones. See [FINDINGS.md](FINDINGS.md) for the honest answer, including what turned out not to work.

The project moved past its first architecture (a custom self-supervised encoder, JEPA) once testing showed a
much simpler approach, raw genome features fed into gradient-boosted trees, actually predicts better. Both are
still in the codebase; FINDINGS.md explains why the simpler one won.

## Setup

```bash
make install   # creates a virtual environment and installs everything
make test      # runs the test suite, no network needed
```

## Running it

```bash
make pull-data              # pull the growth-rate and taxonomy data
make build-features         # build the feature table for the labeled species
make pretrain                # train the encoder (see FINDINGS.md for whether this is worth using)
make finetune-benchmark      # benchmark against gRodon and Phydon
```

The model that makes the current predictions (gradient-boosted trees, not the encoder) is built and run with:

```bash
make predict-all            # ribosomal HMMs, landmark phylogeny, extra species, 16S, gene features and gene content, accuracy table, predictions, report
```

or step by step, see `make help`. Predicting on a list of GEM genomes that don't have a growth-rate label, once the
steps before `predict` have been run:

```bash
python scripts/pipeline/predict_gem_genomes_gbm.py --genome-ids-file <your list>
```

The gene-feature step downloads each GEM genome and takes about an hour for the 9,608 genomes in the HQ and validation lists.

Scripts are split into two folders: `scripts/pipeline/` is the main build-train-evaluate flow above,
`scripts/accuracy/` holds the scripts from the accuracy-improvement work described in ACCURACY_PLAN.md, things
like testing a different model head or trying to patch a specific gap in the training data.

The older script `scripts/pipeline/predict_unlabeled_genomes.py` is kept for reference but its output should not
be used, see `Test on HQ genomes/superseded/README.md` and the 2026-10-04 entries in `research_log.md`.

## What's real and what's approximate

The labeled corpus of species with a measured growth rate is small, 754 genomes from 669 species (bacteria and archaea),
mostly from Madin and colleagues' compilation, plus eight hand-checked strains from lineages it lacks
(`data/curated/`). Ten values from the compilation that are clearly not maximal growth rates are left out, with the
evidence for each in `data/curated/label_exclusions.csv`. Most growth rates are species values, not matched to the exact strain that was sequenced. A genome that is not itself on the reference tree of life gets the position of its species, genus
or family, and the table says which. Predictions are best read as rankings or as rough estimates: a typical error is a
factor of about 2, the predictions are compressed toward the middle so very slow growth is underestimated, and accuracy
falls for genomes with no close labeled relative.

## Where the results are

- `FINDINGS.md` has the actual research findings, written up honestly including the things that didn't pan out.
- `ACCURACY_PLAN.md` tracks the ongoing work to improve the model's accuracy.
- `research_log.md` is a day-by-day log of what was tried and what happened.
- `Test on HQ genomes/` has the predictions for the DOE GEM catalog's high-quality genomes, with a plain-language
  report, charts and the full table. The first two versions are archived in `superseded/` and should not be used.

## A few honest limitations

The rRNA/tRNA gene counts need `barrnap` and `tRNAscan-SE` installed to work, otherwise those columns come back
empty rather than guessed. Codon usage bias is a from-scratch version of the standard method, computed on genes predicted with pyrodigal and ribosomal
proteins found with Pfam models, so it is close to but not identical to gRodon's own numbers (run on the raw genomes it matches the
annotation-based values at a correlation of 0.997). Gene content for energy metabolism and lifestyle (methanogenesis,
sulfate reduction, ammonia oxidation, nitrogen fixation, photosynthesis, transporter density and others) comes from Pfam
family searches on the same predicted genes, so it is a set of presence calls, not a full annotation. The model has been compared with the released gRodon and Phydon R
packages on the same genes and splits (`scripts/accuracy/compare_released_tools.py`, R packages in a project library
`.Rlib`): on the same inputs it has the lower typical error (released gRodon ranks about as well when both get the measured
temperature), and the released tools look better only when they are given the measured growth temperature, which the model
has to predict. Older comparisons in this repository used reimplementations
that were weaker than the real tools. Real 16S extraction from raw
genome sequences is slow, so it only covers part of the large unlabeled genome catalog, not all of it.
