.PHONY: ribosomal-hmms landmarks fetch-16s madin-extension genome-rows labeled-genes gem-genes accuracy-table predict report predict-all install test lint pull-data build-features sanity-check pretrain pretrain-full finetune-benchmark probe necessity-sufficiency pipeline gem-fast gem-slow gem-16s clean help

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
N_PER_CLASS ?= 10

help:
	@echo "GEM-WorldModel Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install            Create .venv (Python 3.11) and install the package + dev deps"
	@echo "  make test               Run the pytest suite (no network calls)"
	@echo "  make lint               Run ruff over src/ and tests/"
	@echo ""
	@echo "Pipeline (live network calls, configured from configs/*.yaml):"
	@echo "  make pull-data          Pull + cross-reference gRodon/GTDB/GEM data"
	@echo "  make build-features     Build the feature table (N_PER_CLASS=10 species/class by default)"
	@echo "  make sanity-check       JEPA sanity check: no collapse, no target-encoder grad leakage"
	@echo "  make pretrain           Self-supervised masked-branch pretraining (labeled corpus only)"
	@echo "  make pretrain-full      Same, but combined with the full GEM MAG corpus (see gem-* below)"
	@echo "  make finetune-benchmark Fine-tune growth-rate head + gRodon/Phydon benchmark"
	@echo "  make probe              Probing + activation intervention (oligotroph/copiotroph)"
	@echo "  make necessity-sufficiency  Necessity/sufficiency masking, regime-specific"
	@echo "  make pipeline           Run the full pipeline in sequence"
	@echo ""
	@echo "  N_PER_CLASS=20 make build-features   Override the sample size"
	@echo ""
	@echo "GEM MAG corpus (real 52,515-genome unlabeled pretraining corpus):"
	@echo "  make gem-fast           Genome traits + phylogeny for ALL 52,515 genomes (no downloads, ~1 min)"
	@echo "  make gem-slow           Real GC content for all 52,515 genomes (streaming download, hours)"
	@echo "  make gem-16s            Real 16S extraction via barrnap on a genome subset (CPU-bound, hours)"

	@echo ""
	@echo "Current prediction workflow (run in this order; see research_log.md 2026-10-12):"
	@echo "  make landmarks          GTDB landmark phylogeny vectors for bacteria and archaea"
	@echo "  make madin-extension    Extra labeled species from Madin's table placed in GTDB"
	@echo "  make fetch-16s          Cache a 16S sequence for every labeled species"
	@echo "  make ribosomal-hmms     Build the ribosomal protein HMM file from Pfam (downloads about 420 MB)"
	@echo "  make labeled-genes      Codon usage and IVYWREL for labeled genomes (pyrodigal + HMMER)"
	@echo "  make gem-genes          The same for the GEM HQ genomes (downloads each genome, hour-scale)"
	@echo "  make accuracy-table     Held-out accuracy and ranges for every feature combination"
	@echo "  make predict            Predict the GEM HQ genomes (stops if inputs drift from training)"
	@echo "  make report             Charts, table and PDF report in 'Test on HQ genomes'"

install:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	.venv/bin/ruff check src/ tests/ scripts/

pull-data:
	$(PYTHON) scripts/pipeline/pull_data.py

build-features:
	$(PYTHON) scripts/pipeline/build_features.py --n-per-class $(N_PER_CLASS)

sanity-check:
	$(PYTHON) scripts/pipeline/pretrain_sanity_check.py

pretrain:
	$(PYTHON) scripts/pipeline/pretrain_labeled.py

pretrain-full:
	$(PYTHON) scripts/pipeline/pretrain_full.py

gem-fast:
	$(PYTHON) scripts/pipeline/gem_fast_features.py

gem-slow:
	$(PYTHON) scripts/pipeline/gem_slow_features.py --workers 10

gem-16s:
	$(PYTHON) scripts/pipeline/gem_slow_features.py --with-16s --workers 8 --limit 5000 \
		--output-name unlabeled_corpus_features_16s.csv

finetune-benchmark:
	$(PYTHON) scripts/pipeline/finetune_benchmark.py

probe:
	$(PYTHON) scripts/pipeline/probe_intervene.py

necessity-sufficiency:
	$(PYTHON) scripts/pipeline/run_necessity_sufficiency.py

pipeline: pull-data build-features sanity-check pretrain finetune-benchmark probe necessity-sufficiency

landmarks:
	$(PYTHON) scripts/accuracy/build_gtdb_landmark_features.py

madin-extension:
	$(PYTHON) scripts/accuracy/build_madin_extension.py

genome-rows: ribosomal-hmms
	$(PYTHON) scripts/accuracy/build_madin_genome_extension.py
	$(PYTHON) scripts/accuracy/build_madin_genome_extension.py --literature

fetch-16s: genome-rows
	$(PYTHON) scripts/accuracy/fetch_labeled_16s.py --features-files features_sample_expanded.csv madin_extension.csv
	$(PYTHON) scripts/accuracy/extract_16s_from_genomes.py

ribosomal-hmms:
	$(PYTHON) scripts/accuracy/build_ribosomal_hmms.py

labeled-genes: ribosomal-hmms
	$(PYTHON) scripts/accuracy/compute_cub_labeled.py
	$(PYTHON) scripts/accuracy/compute_cub_extension.py
	$(PYTHON) scripts/accuracy/compute_lifestyle_labeled.py

gem-genes:
	$(PYTHON) scripts/pipeline/compute_cub_gem.py --ids-file mag_validation_ids.csv hq_genome_ids_all.csv --workers 8
	$(PYTHON) scripts/pipeline/compute_lifestyle_gem.py --ids-file mag_validation_ids.csv hq_genome_ids_all.csv --workers 8

accuracy-table:
	$(PYTHON) scripts/accuracy/evaluate_final_model.py

predict:
	$(PYTHON) scripts/pipeline/predict_gem_genomes_gbm.py

report:
	$(PYTHON) scripts/pipeline/make_report_charts.py
	$(PYTHON) scripts/pipeline/build_report.py

predict-all: ribosomal-hmms landmarks madin-extension genome-rows fetch-16s labeled-genes gem-genes accuracy-table predict report

clean:
	rm -rf checkpoints/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -exec rm -rf {} +
