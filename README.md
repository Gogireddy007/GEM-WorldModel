# GEM-WorldModel

Latent World Models for Genotype-to-Phenotype Prediction: a Joint-Embedding Predictive Architecture (JEPA) for microbial growth dynamics.

This is a cross-species genomic world model. It predicts a masked genomic or phylogenetic branch's latent representation from the others, self-supervised, no growth-rate labels required, then fine-tunes a growth-rate prediction head and runs explainability analysis to answer the main research question: what controls microbial growth rate, and does the answer differ for fast (<5h doubling time) organisms versus slow (≥5h) ones.

It's a masked-branch, snapshot-level architecture. No temporal rollout, per the project's scope after the pivot away from the Keio single-taxon track.

## Architecture

```
Genome + growth-rate corpus (gRodon/Madin + GTDB + GEM MAGs)
         │
         ├── Genomic traits (CUB, rRNA/tRNA, size, GC, reg. genes) ──┐
         ├── GTDB-distance embedding (primary phylogeny signal) ─────┼── Branch masking
         └── 16S rRNA embedding (baseline, for comparison) ──────────┘        │
                                                                    ┌──────────┴──────────┐
                                                              Context encoder      Target encoder
                                                                  E_θ                E_ξ (EMA)
                                                                    │                    │
                                                              Predictor P_φ  ──►  Latent loss D(ŝ, s)
                                                                    │
                                                        (joint repr., all branches unmasked)
                                                                    │
                                                    ┌───────────────┼───────────────┐
                                              Growth-rate head   Probing +      Necessity/
                                             (fine-tuned)      intervention    sufficiency
                                                    │           (oligotroph/     masking
                                              Benchmark vs.     copiotroph)    (regime-split)
                                             gRodon & Phydon
```

Exact branch/encoder/predictor dimensions live in `configs/model.yaml`.

## Data sources

The gRodon2/Madin (2020) growth-rate corpus is pulled live from `jlw-ecoevo/gRodon2`'s GitHub `.rda` files and parsed with `pyreadr`, so no R runtime is needed.

GTDB (`data.gtdb.ecogenomic.org`) supplies the bac120 reference tree and taxonomy for the GTDB-distance phylogeny branch. Note that the taxonomy table and the tree are not the same thing: the taxonomy table lists every genome GTDB has ever classified, but the tree only has representative genomes as tips. A genome can have GTDB taxonomy without being placeable on the tree.

NCBI provides genome assemblies and annotated CDS per accession, plus 16S rRNA sequences via Entrez, for feature extraction.

The DOE NERSC GEM portal (`portal.nersc.gov/GEM`) has 52,515 MAGs, used as the large unlabeled corpus for self-supervised pretraining, since most of them have no measured growth rate. GEM ships its own phylogenetic tree too (`multi_marker.rooted.tree`, 43,979 OTUs), which is what these genomes actually get placed on, since their JGI-style IDs don't match NCBI/GTDB accessions. A Dropbox folder of the same genomes is wired up as an alternative source in `data/gem_portal.py`.

None of this data is checked into the repo (see `.gitignore`). `data/raw/` and `data/processed/` get populated by running the scripts below.

## Setup

```bash
make install   # creates .venv (Python 3.11 - torch doesn't have wheels for 3.14 yet)
make test      # pytest, no network calls
```

## Running the pipeline

Each script covers one stage and can be run standalone or chained with `make pipeline`:

```bash
make pull-data                          # pull + cross-reference gRodon/GTDB/GEM data
make build-features N_PER_CLASS=122     # build the feature table for the labeled corpus
make sanity-check                       # JEPA sanity check (no collapse, no grad leakage to target encoder)
make pretrain                           # self-supervised masked-branch pretraining
make finetune-benchmark                 # fine-tune growth-rate head + gRodon/Phydon benchmark
make probe                              # probing + activation intervention
make necessity-sufficiency              # necessity/sufficiency masking, split by doubling-time regime
```

For the GEM MAG corpus specifically:

```bash
make gem-fast    # genome traits + phylogeny for all 52,515 genomes, no downloads needed, ~1 min
make gem-slow    # real GC content, streaming download, several hours at full scale
make gem-16s     # real 16S extraction via barrnap on a genome subset, CPU-bound, slow
```

A note on scale: the actual usable labeled corpus turned out to be much smaller than it first looked. Of the roughly 87,000 gRodon/Madin accessions, 93% have GTDB taxonomy, but only about 0.3% (271 accessions, 175 species) are real tips on the GTDB reference tree. That 175-species number is the true ceiling for anything using the phylogeny branch, and `build_features.py`'s stratified sample now reflects that.

## Known limitations

rRNA and tRNA gene counts need `barrnap`/`tRNAscan-SE` on the PATH. If they're not installed, `features/genome_traits.py` returns NaN for those columns instead of guessing. Install the tools and the same code starts producing real counts.

CUB (codon usage bias) is a from-scratch reimplementation of the MILC statistic in `features/cub.py`, not a call into gRodon's actual R package, so expect values that are close but not bitwise identical to the original.

The gRodon and Phydon baselines in `training/baselines.py` are reimplementations of each method's feature set and model class, refit on our own train split. They're not the original papers' published coefficients, which aren't available outside their R packages anyway and wouldn't be a fair same-split comparison even if they were.

16S sequences come from NCBI by organism name (one representative record per species) rather than being extracted directly from genome assemblies for the labeled corpus. For the GEM MAG corpus, real 16S extraction via barrnap is genuinely slow (measured throughput was well under 1 genome/second even with several workers), so full coverage of all 52,515 genomes isn't practical in one sitting. `gem_slow_features.py --with-16s` processes a bounded subset instead.

The oligotroph/copiotroph label used in probing has no bundled literature-curated source. `eval/probing.py:heuristic_trophic_label` is an explicitly flagged proxy based on genome size and rRNA copy number, meant for testing the pipeline, not for drawing real conclusions. Swap in an actual ecological label before treating those probing results as meaningful. The same caveat applies to the CUB/oligotroph relationship generally: oligotrophs tend to show weak codon usage bias in the first place, so a probe that finds CUB-correlated structure predicting oligotroph status risks confirming something that's circular by construction rather than discovering it.

## Layout

```
configs/            data.yaml, features.yaml, model.yaml, train.yaml
src/gem_worldmodel/
  data/              acquisition: grodon.py, gtdb.py, gem_tree.py, gem_portal.py, ncbi_genomes.py,
                     consolidate.py, validate.py
  features/          cub.py, genome_traits.py, phylogeny.py, rrna16s.py, temperature.py, build.py
  models/            encoders.py, masking.py, predictor.py, losses.py, heads.py, jepa.py
  training/          dataset.py, pretrain.py, finetune.py, baselines.py
  eval/              benchmark.py, probing.py, intervention.py, necessity_sufficiency.py
scripts/             pull_data.py, build_features.py, pretrain_sanity_check.py, pretrain_labeled.py,
                     pretrain_full.py, finetune_benchmark.py, probe_intervene.py,
                     run_necessity_sufficiency.py, gem_fast_features.py, gem_slow_features.py
tests/               unit + smoke tests, no network calls
```
