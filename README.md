# GEM-WorldModel

Code and data for the paper *How fast do uncultured microbes grow?*

Most microbes are known only from metagenome-assembled genomes, so their growth rates have to be inferred from sequence.
This repository predicts three quantities from a genome alone:
- the maximal doubling time;
- the optimal growth temperature;
- a doubling time standardized to 20 °C.

Every prediction is evaluated with whole species, genera, families or orders held out. The trained model is applied to the
9,139 high-quality genomes of the Genomes from Earth's Microbiomes (GEM) catalog (Nayfach et al. 2021). Each predicted
value is released with its expected error and an 80% range.

Version 1.25.0 (2026-11-16). License: Apache-2.0.

## Main results

| Evaluation | Typical fold error |
|---|---|
| Labeled relative in the same genus | 1.75 |
| No labeled relative in the same order | 2.08 |
| New species, without the measured growth temperature | 1.76 |
| gRodon on new species, given the measured temperature | 1.92 |
| Phydon on new species, given the measured temperature | 1.85 |
| External test, 48 species absent from both training compilations | 2.53 (taxonomy alone: 3.65) |

Typical fold error is the median of measured/predicted or its inverse.

For new orders, the model trails gRodon when gRodon is given the measured temperature. On the external test, the 80% ranges
contain only 67% of the measured values, and slow growers are underestimated. [FINDINGS.md](FINDINGS.md) discusses these
results, including the approaches that did not work.

## Method

Each genome is described by three groups of inputs:
1. **Genome traits:** GC content, genome size, rRNA and tRNA gene counts, and codon usage bias of the ribosomal proteins.
2. **Position on the GTDB tree of life.**
3. **Gene content:** presence calls for energy-metabolism and lifestyle families, such as methanogenesis, sulfate reduction,
   nitrogen fixation, photosynthesis and transporter density. The 16S rRNA gene is also used, where it can be recovered.

A gradient-boosted tree model is trained on these inputs, on 754 genomes from 669 species with a measured growth rate.
- Most labels come from the compilation of Madin and colleagues.
- Eight hand-checked strains are added from lineages it lacks (`data/curated/`).
- Ten values that are clearly not maximal growth rates are excluded; the evidence for each is in
  `data/curated/label_exclusions.csv`.

The first version of the project used a self-supervised encoder (JEPA). Under the same held-out tests, raw genome features
with gradient-boosted trees predicted better, so the encoder is not used for the released predictions. Its code is kept in
`src/gem_worldmodel/models/`, and FINDINGS.md compares the two.

## Installation

Requirements:
- Python 3.11;
- for the full pipeline, R 4.6.1, HMMER 3.4, barrnap 0.9 and tRNAscan-SE 2.0.13;
- a LaTeX installation with `latexmk` to build the PDFs.

```bash
git clone https://github.com/Gogireddy007/GEM-WorldModel.git
cd GEM-WorldModel
make install        # virtual environment with the exact package versions in requirements-lock.txt
make r-packages     # R packages, including gRodon 2.7.3 and Phydon 1.1.0, in a project library .Rlib
make test           # unit tests; no network access needed
```

## Reproducing the paper

[REPRODUCE.md](REPRODUCE.md) gives the full procedure. There are two routes:

| Command | Starts from | Time |
|---|---|---|
| `make paper` | public raw data | about a day on ten cores, about 15 GB of disk |
| `make paper-from-archive` | `data.zip` attached to release [v1.25.0](https://github.com/Gogireddy007/GEM-WorldModel/releases/tag/v1.25.0) | a few minutes |

Every download is pinned to the version the paper used:
- GTDB release 232;
- Pfam 38.2;
- the Madin et al. and gRodon repositories at fixed commits;
- versioned NCBI assemblies.

`make verify-inputs` checks the downloaded files against the SHA-256 checksums in `data/input_checksums.tsv`.

Every number in the manuscripts is a macro in `paper/numbers.tex`, written from the result files by
`paper/scripts/make_numbers.py`. `make paper-check` recomputes 121 numbers and claims independently from the per-genome
results, and checks each cited DOI and external label against its source.

## Predictions for the GEM catalog

The released table is `release/gem_growth_predictions.csv`, with one row per genome. Its columns are described in
`release/COLUMNS.md`. Each row carries:
- the three predictions;
- the 80% range and typical fold error at the genome's novelty level;
- the closest rank it shares with a labeled species;
- the model inputs that were available for it.

Of the 9,139 genomes, 38% have no labeled relative in their order.

To predict other genomes from the GEM catalog, run the steps before `predict` (see `make help`), then:

```bash
python scripts/pipeline/predict_gem_genomes_gbm.py --genome-ids-file <list of GEM genome IDs>
```

## GrowthBench

The labeled genomes, the lineage-held-out splits and the scores in the paper are available through a small interface. A new
model can be scored on the same folds as the paper, next to the required taxonomy baseline:

```python
from gem_worldmodel import benchmark as gb

data = gb.load()   # data.table: taxonomy and targets; data.X: model inputs

def fit_predict(train, test, X_train, X_test):
    return my_model.fit(X_train, train["y"]).predict(X_test)   # natural log of doubling time in hours

results = gb.evaluate(fit_predict, data)   # species, genus, family and order held out; 3 seeds
print(results.groupby(["held_out", "method"])[["typical error", "Spearman"]].mean())
```

The splits are identical to those behind every result in the paper; `tests/test_benchmark.py` checks this fold by fold.
Results that use the measured growth temperature as an input should be reported separately.

## Repository layout

| Path | Contents |
|---|---|
| `src/gem_worldmodel/` | Python package: data download, features, models, evaluation, GrowthBench |
| `scripts/pipeline/` | the main pipeline: download, features, training, prediction, input verification |
| `scripts/accuracy/` | accuracy studies (see ACCURACY_PLAN.md) and the comparison with gRodon and Phydon |
| `scripts/r/` | R package installation and lock file |
| `configs/` | data sources and model settings |
| `data/curated/` | literature labels, label exclusions and the external test set |
| `data/fixed/` | inputs that cannot be regenerated identically; the reason for each is in its README |
| `paper/` | ISMB manuscript (Bioinformatics template) and supplement |
| `paper_neurips/` | NeurIPS Datasets and Benchmarks manuscript (GrowthBench) |
| `release/` | released GEM prediction table and column descriptions |
| `tests/` | unit and smoke tests |

`paper/scripts/make_archive.py` assembles an archive in `dist/zenodo/` for deposit. It holds the code, the derived tables and
result files, the release table, the PDFs and a SHA-256 manifest. Both manuscripts list anonymous authors for review.

## Limitations

- **Small labeled set.** The labeled set is small, and most growth rates are species values, not measurements of the
  sequenced strain.
- **Approximate tree position.** A genome that is not on the reference tree is placed at its species, genus or family, and the
  table records which.
- **Compressed predictions.** Predictions are pulled toward the middle of the range, so very slow growth is underestimated.
  They are best read as rankings or as estimates within a factor of about 2.
- **Lower accuracy for novel lineages.** Accuracy falls for genomes with no close labeled relative.
- **Codon usage bias is reimplemented.** It is computed on genes predicted with pyrodigal, with ribosomal proteins found with
  Pfam models. It is close to gRodon's values but not identical: the correlation with the annotation-based values is 0.997.
- **Gene content is presence calls.** It comes from Pfam family searches, not a full functional annotation.
- **rRNA and tRNA counts need external tools.** They require barrnap and tRNAscan-SE; without them, those columns are left
  empty rather than imputed.
- **Not significant per genome.** Without the measured temperature, the model's typical error is lower than that of gRodon or
  Phydon given it. Per genome, though, the difference is not significant: p = 0.07 against gRodon and 0.11 against Phydon
  (`scripts/accuracy/pressure_tests.py`).
- **Partial 16S coverage.** 16S extraction from raw genomes is slow, so it covers only part of the GEM catalog.

## Further documentation

- [REPRODUCE.md](REPRODUCE.md): rebuilding every result from scratch.
- [FINDINGS.md](FINDINGS.md): research findings, including negative results.
- [CHANGELOG.md](CHANGELOG.md): changes in each version.
- `research_log.md`: dated record of each experiment and its numbers.
- `ACCURACY_PLAN.md`: the accuracy-improvement work.
- `Test on HQ genomes/`: a plain-language report with charts on the predictions for the GEM high-quality genomes. The
  versions in its `superseded/` folder should not be used.
