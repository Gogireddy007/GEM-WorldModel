.PHONY: install test lint pull-data build-features sanity-check pretrain pretrain-full finetune-benchmark probe necessity-sufficiency pipeline gem-fast gem-slow gem-16s clean help

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
	$(PYTHON) scripts/pull_data.py

build-features:
	$(PYTHON) scripts/build_features.py --n-per-class $(N_PER_CLASS)

sanity-check:
	$(PYTHON) scripts/pretrain_sanity_check.py

pretrain:
	$(PYTHON) scripts/pretrain_labeled.py

pretrain-full:
	$(PYTHON) scripts/pretrain_full.py

gem-fast:
	$(PYTHON) scripts/gem_fast_features.py

gem-slow:
	$(PYTHON) scripts/gem_slow_features.py --workers 10

gem-16s:
	$(PYTHON) scripts/gem_slow_features.py --with-16s --workers 8 --limit 5000 \
		--output-name unlabeled_corpus_features_16s.csv

finetune-benchmark:
	$(PYTHON) scripts/finetune_benchmark.py

probe:
	$(PYTHON) scripts/probe_intervene.py

necessity-sufficiency:
	$(PYTHON) scripts/run_necessity_sufficiency.py

pipeline: pull-data build-features sanity-check pretrain finetune-benchmark probe necessity-sufficiency

clean:
	rm -rf checkpoints/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -exec rm -rf {} +
