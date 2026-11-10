# Build the manuscript: make (numbers, tables and figures from the result files, then the PDFs). Run from paper/.

all: numbers supplement.pdf main.pdf audit

numbers:
	cd .. && .venv/bin/python paper/scripts/make_numbers.py
	cd .. && .venv/bin/python paper/scripts/make_supp_tables.py
	cd .. && .venv/bin/python paper/scripts/make_tables.py
	cd .. && .venv/bin/python paper/scripts/figures.py

figures/fig1_overview_standalone.pdf: figures/fig1_overview_standalone.tex numbers.tex
	cd figures && latexmk -pdf -interaction=nonstopmode -halt-on-error fig1_overview_standalone.tex

main.pdf: main.tex gem-oup.bst sections/*.tex tables/*.tex figures/* figures/fig1_overview_standalone.pdf refs.bib numbers.tex supplement.pdf
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

supplement.pdf: supplement.tex supplement/*.tex refs.bib numbers.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex

# rules of PLAN.md: macros, typed numbers, style, page limit
audit: main.pdf
	cd .. && .venv/bin/python paper/scripts/audit_paper.py

.PHONY: all numbers audit clean

clean:
	latexmk -C
