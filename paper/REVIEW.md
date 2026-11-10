# Internal review (phase 6 and pressure test, 2026-11-01)

Read against the ISMB 2026 proceedings rules (OUP *Bioinformatics* template, 9 pages including references, structured
abstract of at most 250 words, alt text for every image, public code, not double-blind) and against the reviewing
questions: is the contribution new and significant, is the method sound, are the claims supported, is it clear, can it be
reproduced. The 2027 call is not out yet; recheck these rules when it is.

## Format

| Rule | Status |
|---|---|
| OUP template | `oup-authoring-template` class from TeX Live 2026, contemporary large layout, as in OUP's sample; font size and spacing untouched |
| 9 pages with references | 9 (`make audit` fails above 9) |
| Structured abstract, 250 words | 226 words from the PDF, headings and URL included |
| Alt text | all five figures (`alt=` key; Figure 1 is built as a standalone PDF so it can carry one) |
| Public code | github.com/Gogireddy007/GEM-WorldModel is public; the paper work must be pushed before submission |
| Supplementary material | separate file, `supplement.pdf`; ISMB discourages it, so the main text stands without it |

References use `gem-oup.bst`, OUP's `oup-abbrvnat.bst` with three local changes (more than ten authors shortened to three
and et al., no DOIs, ISSNs or ISBNs, URLs only for software), to fit the page limit.

## Claims

Every number in the text is a macro from `numbers.tex` (`audit_paper.py` finds no typed result numbers in the prose; the
allow list in the script gives the reason for each definitional number). Every claim is in `CLAIMS.md` with its evidence.
Rechecked by hand in this pass: the eigenvector robustness of the transporter result (10 and 30 eigenvectors), the
regulator effect depending on Pseudomonadota (p 0.92 without them), the across-species energy effects named in Results,
the two orders that hold 34 of the 39 slow GEM genomes, and every alt text against the figure data (one draft alt text
said all energy intervals cross one; F-type ATP synthase does not, so it was corrected).

## What a reviewer is likely to raise, and where it is answered

1. **The learner is standard.** The paper does not claim a new learner. The contributions are the lineage-held-out evaluation
   with per-prediction errors, the label audit, the comparison with the released tools on equal inputs, the MAG test and the
   GEM resource (Introduction).
2. **Is the comparison with gRodon and Phydon fair?** Same genes, folds and species; Phydon's training table is stripped of
   each fold's species; the temperature advantage of the released tools is stated and shown both ways; their remaining
   advantage (parameters fitted on data that include our species) is stated. It was run with species held out only, now
   stated in Limitations.
3. **Three seeds for the headline.** Bars span the seeds (Figure 2); the ablations use ten seeds and say that seeds reshuffle
   the same genomes, so p measures consistency, not replication.
4. **Slow growth.** Stated plainly: ranked (AUC) but rarely predicted, none above a day for new orders; the 39 GEM calls are
   a short confident list.
5. **Label errors remain.** Stated; the audit lowers the error rate but cannot certify single-study values.
6. **Negative energy result could be underpowered.** Stated: few labeled organisms on the thinnest energy budgets, so it is
   evidence against the simplest form among cultured organisms only.
7. **Novelty levels of GEM genomes use an older GTDB release.** Stated as a worst case (Methods, Limitations).

## Left for the authors

- Push the paper work to the public repository, and deposit the archive (`paper/scripts/make_archive.py`) at Zenodo; the
  availability section says it will be deposited with the final version.
- Authors: both manuscripts list "Anonymous Authors" with no affiliation, email or repository link (2026-11-09); restore
  the author block for the camera-ready version.
- Recheck the rules when the ISMB 2027 call is posted, and use the journal's own Overleaf copy of the template if it differs
  from the TeX Live class.

## Pressure test (2026-11-01)

New script `scripts/accuracy/pressure_tests.py` (run by `make paper-results`), on the headline's own folds:

| Test | Result | What changed in the paper |
|---|---|---|
| Bootstrap over species groups (2,000 draws) | 1.75x (1.68 to 1.85) with species held out, 2.08x (1.99 to 2.18) with orders held out; the intervals do not overlap | intervals reported in Results |
| Median of the closest labeled relatives, same folds | species held out: 1.81x, difference to the model -0.19 to 0.04 (includes zero), Spearman 0.66 against 0.74; families: 2.59x, orders: 2.45x, both clearly worse than the model | new Results paragraph, dotted line in Figure 2, Discussion now opens with where the model adds most; supplementary table of all levels |
| Labels shuffled across genomes | 2.91x and Spearman 0.00 (species held out), no better than the median (2.73x) | negative control reported |
| Paired per-genome test against released tools | given the temperature: ours better per genome (p 1.5e-6 against gRodon, 3.9e-4 against Phydon); without it: lower typical error but per genome not significant (349/680, p 0.07; 347/665, p 0.11) | the claim "lower error than the tools given the temperature" was weakened to "as accurate as" in abstract, introduction and Results; Figure 3 became Table 2 with the paired wins; README updated |
| MAG bootstrap over species | 1.86x (1.60 to 2.20) against isolates 1.70x; difference -0.15 to 0.33 includes zero | interval reported |
| Energy claim wording | a sub-analysis among the slowest groups gives sulfate reduction an unadjusted p of 0.02 | the Discussion now restricts the claim to adjusted p in the main analysis and the three tabulated checks |

Style: Methods rewritten with notation and equations (targets, metrics, the 80% range, the model, the placement
augmentation, PGLS) and two algorithms (the study-level label check and the nested lineage-held-out evaluation), all
checked against `screen_label_sources.py`, `tune_deployed_model.py` and `evaluate_paper_headline.py`. The layout stays the
journal's (the Overleaf Bioinformatics template uses the same contemporary large layout; ISMB forbids changes to font size
or spacing).

## Senior-panel review and response (2026-11-01)

A three-pass review in the style of a NeurIPS or ICLR panel scored the ISMB paper 44/100 for an ML main track (no new
learning method) and about 70/100 for ISMB. Its points and what was done:

| Panel point | Response |
|---|---|
| 80% ranges calibrated only on average; archaea 73-74% with orders held out | ranges now set by novelty level and domain (`calibrate_ranges.py`, cross-fitted over genera): archaea at least 78% at every level; released table uses them |
| Slow growers outside the ranges (31-45%) and predicted 4.5- to 6.2-fold too fast, not reported | reported in both papers; no range built from the inputs can target them, so a calibrated probability of growing slower than a day is released (AUC 0.81-0.84, calibration error at most 0.012, top fifth 22-24% slow against 8%) |
| Archaea worse than bacteria, not reported | reported (2.43 against 2.02 with orders held out) |
| Bootstrap over species too narrow (species of a genus are correlated) | intervals now resample genera, or the held-out families or orders; intervals widen, conclusions unchanged |
| No learned-representation baseline | ESM-2 (35M) embeddings of every ribosomal protein and 100 sampled proteins per genome, evaluated on the same folds as a ridge probe, trees on principal components, and combined with the features (`embed_proteins_esm.py`, `evaluate_esm_baseline.py`) |
| No novel method for an ML main track | not fixable honestly by rewriting; a separate NeurIPS Datasets and Benchmarks version (`paper_neurips/`) presents the work as an audited benchmark with a tested interface (`gem_worldmodel.benchmark`) |
