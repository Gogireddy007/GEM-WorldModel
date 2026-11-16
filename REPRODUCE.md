# Rebuilding the paper from scratch

Everything needed is in this repository or in public sources at fixed versions. There are two routes:

- **From scratch.** Download the raw data and recompute everything. This takes about a day on ten cores and about 15 GB of disk.
- **From the data archive.** Start from the derived tables attached to the GitHub release. This takes several hours and needs no genome downloads.

## What you need

- macOS or Linux, 10 cores and 16 GB of memory are enough.
- Python 3.11. Exact package versions are in `requirements-lock.txt`.
- R 4.6.1. Exact package versions are in `scripts/r/R_packages.lock.tsv`; gRodon 2.7.3 and Phydon 1.1.0 are installed at fixed commits.
- Command-line tools (exact versions in `tools-lock.tsv`):
  - HMMER 3.4 (`hmmsearch`)
  - barrnap 0.9
  - tRNAscan-SE 2.0.13
  - a LaTeX installation with `latexmk` (TeX Live 2026 was used)

  On macOS:

```bash
brew install python@3.11 r hmmer barrnap trnascan-se
```

## Setup

```bash
git clone https://github.com/Gogireddy007/GEM-WorldModel.git
cd GEM-WorldModel
make install        # .venv with the exact Python packages
make r-packages     # .Rlib with the exact R packages; prints any package whose version differs
make test           # unit tests, no network
```

## Route 1: from scratch

```bash
make paper          # about a day; downloads about 10 GB
make verify-inputs  # every downloaded file against data/input_checksums.tsv
make paper-check    # recomputes 121 numbers and claims from the per-genome files, checks references and labels
```

`make paper` runs, in order:
1. The downloads.
2. The labeled feature tables.
3. The GEM genome features.
4. Gene content and codon usage for every genome.
5. Model tuning and every evaluation in the paper.
6. The GEM predictions.
7. The numbers, tables, figures and both PDFs.

The PDFs are `paper/main.pdf`, `paper/supplement.pdf` and `paper_neurips/main.pdf`. Every number in them is a macro in `paper/numbers.tex`, written from the result files by `paper/scripts/make_numbers.py`.

### Pinned sources

Every download is pinned, so a rerun next year reads the same files. `data/input_checksums.tsv` lists the SHA-256 of all 1,323 downloaded files.

| Source | Version used | Where it is set |
|---|---|---|
| gRodon growth-rate tables (Madin et al.) | gRodon2 commit `4b13053` | `configs/data.yaml` |
| GTDB trees and taxonomy | release 232 | `configs/data.yaml` |
| Madin et al. trait tables | commit `396dcd4` | `madin_traits.py`, `screen_labels.py` |
| GEM catalog metadata and tree | NERSC portal (static since 2020) | `configs/data.yaml`, `gem_tree.py` |
| Pfam-A HMMs | release 38.2 | `build_ribosomal_hmms.py` |
| NCBI genome assemblies | versioned accessions (e.g. `GCF_000005845.2`) | `data/input_checksums.tsv` |
| gRodon, Phydon (R) | 2.7.3 at `4b13053`, 1.1.0 at `321bb76` | `scripts/r/install_packages.R` |
| ESM-2 protein model | `esm2_t12_35M_UR50D` (fair-esm 2.0.0) | `embed_proteins_esm.py` |

A few inputs are part of the method and are kept in `data/fixed/`, with the reason for each in `data/fixed/README.md`:
- the GEM genome lists;
- the 16S sequence chosen for each labeled species;
- the 16 reference 16S genes.

They are kept because the step that first made them depended on an NCBI search, whose results change over time.

### What can still differ

- **NCBI suppressing an assembly.** NCBI occasionally suppresses an assembly. The code then falls back to the GenBank copy of the same assembly. `make verify-inputs` reports any file that differs.
- **Failed downloads of GEM genomes.** These are fetched one at a time, and a genome whose download fails after retries gets no gene features. The run log lists them, and rerunning the same step retries them.
- **Floating-point noise.** Model training is seeded. On a different CPU or BLAS library, the last digit of a few numbers can move; the paper rounds to two decimals.

## Route 2: from the data archive

The GitHub release [v1.25.0](https://github.com/Gogireddy007/GEM-WorldModel/releases/tag/v1.25.0) attaches `data.zip`, which holds:
- the derived feature tables;
- the GEM genome features;
- the result file behind every number, table and figure.

`MANIFEST.tsv` lists the SHA-256 of every file in it. To use it, inside the repository:

```bash
make install
curl -LO https://github.com/Gogireddy007/GEM-WorldModel/releases/download/v1.25.0/data.zip
unzip data.zip        # fills data/processed and data/raw
make paper-from-archive
```

R is not needed for this route.

This takes a few minutes and needs no genome downloads. It does three things:
1. Retrains the deployed model.
2. Predicts the GEM genomes again.
3. Rebuilds the numbers, tables, figures and both PDFs from the archived result files.

`paper/numbers.tex` and `release/gem_growth_predictions.csv` come out identical to the ones in the repository.

The evaluations themselves (held-out accuracy, the comparison with gRodon and Phydon, the external test) need the genomes, so they run only in route 1.
