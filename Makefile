.PHONY: install test lint week1 week2 week3 week4 week4-full week5 week6 week7 pipeline gem-fast gem-slow gem-16s clean help

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
N_PER_CLASS ?= 10

help:
	@echo "GEM-WorldModel — Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Create .venv (Python 3.11) and install the package + dev deps"
	@echo "  make test             Run the pytest suite (no network calls)"
	@echo "  make lint             Run ruff over src/ and tests/"
	@echo ""
	@echo "Pipeline (each week's script from configs/*.yaml, live network calls):"
	@echo "  make week1            Pull + cross-reference gRodon/GTDB/GEM data"
	@echo "  make week2            Build the feature table (N_PER_CLASS=10 species/class by default)"
	@echo "  make week3            JEPA sanity check: no collapse, no target-encoder grad leakage"
	@echo "  make week4            Self-supervised masked-branch pretraining (labeled corpus only)"
	@echo "  make week4-full       Same, but combined with the full GEM MAG corpus (see gem-* below)"
	@echo "  make week5            Fine-tune growth-rate head + gRodon/Phydon benchmark"
	@echo "  make week6            Probing + activation intervention (oligotroph/copiotroph)"
	@echo "  make week7            Necessity/sufficiency masking, regime-specific"
	@echo "  make pipeline         Run week1 through week7 in sequence"
	@echo ""
	@echo "  N_PER_CLASS=20 make week2   Override the Week 2 sample size"
	@echo ""
	@echo "GEM MAG corpus (real 52,515-genome unlabeled pretraining corpus):"
	@echo "  make gem-fast          Genome traits + phylogeny for ALL 52,515 genomes (no downloads, ~1 min)"
	@echo "  make gem-slow          Real GC content for all 52,515 genomes (streaming download, hours)"
	@echo "  make gem-16s           Real 16S extraction via barrnap on a genome subset (CPU-bound, hours)"

install:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	.venv/bin/ruff check src/ tests/ scripts/

week1:
	$(PYTHON) scripts/week1_pull_data.py

week2:
	$(PYTHON) scripts/week2_build_features.py --n-per-class $(N_PER_CLASS)

week3:
	$(PYTHON) scripts/week3_unit_check.py

week4:
	$(PYTHON) scripts/week4_pretrain.py

week4-full:
	$(PYTHON) scripts/week4_pretrain_full.py

gem-fast:
	$(PYTHON) scripts/gem_fast_features.py

gem-slow:
	$(PYTHON) scripts/gem_slow_features.py --workers 10

gem-16s:
	$(PYTHON) scripts/gem_slow_features.py --with-16s --workers 8 --limit 5000 \
		--output-name unlabeled_corpus_features_16s.csv

week5:
	$(PYTHON) scripts/week5_finetune_benchmark.py

week6:
	$(PYTHON) scripts/week6_probe_intervene.py

week7:
	$(PYTHON) scripts/week7_necessity_sufficiency.py

pipeline: week1 week2 week3 week4 week5 week6 week7

clean:
	rm -rf checkpoints/ .pytest_cache/ .ruff_cache/
	find . -name __pycache__ -exec rm -rf {} +
