# NeurIPS Datasets and Benchmarks version. Run from paper_neurips/ after the ISMB paper is built (its numbers and figures).

all: tables main.pdf

tables:
	cd .. && .venv/bin/python paper_neurips/make_neurips_tables.py

main.pdf: main.tex datasheet.tex appendix.tex checklist.tex tables/*.tex refs.bib ../paper/numbers.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

.PHONY: all tables clean

clean:
	latexmk -C
