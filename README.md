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

Predicting on new genomes that don't have a growth-rate label:

```bash
python scripts/pipeline/predict_unlabeled_genomes.py --genome-ids-file <your list>
```

Scripts are split into two folders: `scripts/pipeline/` is the main build-train-evaluate flow above,
`scripts/accuracy/` holds the scripts from the accuracy-improvement work described in ACCURACY_PLAN.md, things
like testing a different model head or trying to patch a specific gap in the training data.

## What's real and what's approximate

The labeled corpus of species with a measured growth rate is small, a few hundred species. Some of them are
placed exactly on the reference tree of life; others don't have an exact spot, so their position is approximated
from a close relative instead. Every row in the data says which kind it is, so nothing gets treated as more
certain than it actually is.

## Where the results are

- `FINDINGS.md` has the actual research findings, written up honestly including the things that didn't pan out.
- `ACCURACY_PLAN.md` tracks the ongoing work to improve the model's accuracy.
- `research_log.md` is a day-by-day log of what was tried and what happened.
- `Test on HQ genomes/` has a real prediction run on the DOE GEM catalog's high-quality genomes, written up as a
  plain report with the actual numbers and charts.

## A few honest limitations

The rRNA/tRNA gene counts need `barrnap` and `tRNAscan-SE` installed to work, otherwise those columns come back
empty rather than guessed. The codon usage calculation is a from-scratch version of the standard method, close
to but not identical to gRodon's own numbers. The gRodon and Phydon comparisons in this project are
reimplementations refit on the same data split, not the original published models. Real 16S extraction from raw
genome sequences is slow, so it only covers part of the large unlabeled genome catalog, not all of it.
