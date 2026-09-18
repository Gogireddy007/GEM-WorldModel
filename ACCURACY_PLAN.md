# Accuracy improvement plan

Started version 1.0, 2026-09-17. Everything before this point stays as it is,
this file tracks the next phase of work only: making the model's actual
predictions more accurate, on any genome it is given, not just adding more
coverage or more corpus size.

Every task here gets checked off only once it has actually been run and the
result has been written into research_log.md and, if it changes a real
number, FINDINGS.md too. A task that does not help gets reported as not
helping. That has been true for every result in this project so far and it
stays true here.

## Phase 0, quick checks

- [x] Check whether the codon usage feature we compute ourselves carries as
      much signal on its own as gRodon's own formula does, tested on the
      same held out species. Done 2026-09-17, see research_log.md. Same
      feature, not a feature problem. gRodon's plain linear regression on
      it gets R2 -0.032, our model using the whole genomic_traits branch
      gets R2 -0.467. The gap is in how our encoder and head handle that
      branch, not in the feature itself.
- [x] Look across the whole training corpus for other places, besides the
      high GC, small genome region already found, where every labeled
      example comes from one narrow kind of organism. Done 2026-09-17, see
      research_log.md. Found a second one: genomes with medium GC content
      and a large genome, 5.2 to 7.5 million base pairs, are 83 percent
      Bacillus in the training set. Any real genome landing there that is
      not Bacillus-like will get a prediction shaped mostly by Bacillus
      biology. Also confirmed the earlier thermophile finding is broader
      than just the genus Thermus, 82 percent of genomes in that same high
      GC, small genome region grow at extreme temperatures generally.
- [x] Try a stronger regression head on top of the same standardized
      features (for example gradient boosted trees) to see whether the
      growth rate head itself, not the encoder, is leaving accuracy on the
      table. Done 2026-09-17, see research_log.md and FINDINGS.md's new
      "A stronger head changes the picture" section. Result, corrected
      after running the proper control: the gain is not from the encoder,
      it is entirely from the head. Raw features with no encoder at all,
      fed into gradient boosted trees, beats the encoder version in all
      seven seeds, and beats gRodon outright on R2 and Spearman in all
      seven seeds, and beats Phydon on R2 in all seven seeds too. This is
      the best result the project has produced. Two things it does not
      fix: the fast growth regime is still very poor, and the tree
      model's own feature importance does not fully agree with the
      necessity and sufficiency finding that 16S dominates, that
      disagreement is real and still open.

## Phase 0.5, follow up on what Phase 0 actually found

- [x] Look into why the fast growth regime is still poor under raw features
      plus gradient boosted trees. Done 2026-09-17, see research_log.md.
      72 percent of fast growers get predicted slower than they really are.
      The whole corpus's median doubling time is 7 hours but the fast
      regime's true median is 1.71 hours, and a model minimizing squared
      error over that whole skewed range gets pulled toward the middle and
      compresses fast growers upward. Not a feature or head problem, a
      training-loss and label-distribution problem. Needs a loss change or
      reweighting to fix, not more tinkering with the current setup.
- [x] Look into why gradient boosted trees' own feature importance does not
      match the necessity and sufficiency finding that 16S dominates. Done
      2026-09-17, see research_log.md. Refit the trees separately on just
      slow growers and just fast growers instead of the whole corpus at
      once. Within slow growers, rrna16s importance rises from 0.24 to
      0.41, same direction as the encoder-based finding, it just does not
      come out as clearly first, gtdb_distance edges it out at 0.47. Real,
      partial agreement, not a full match.
- [x] Now that raw features plus gradient boosted trees is the best result,
      rerun the necessity and sufficiency style check using this model
      instead of the JEPA pipeline, using the tree's own feature importance
      per branch, split by fast and slow regime the same way as before.
      Done as part of the item above, same numbers.

## Phase 1, combine the best model with the existing tools

- [x] Build a proper cross validated blend of raw features plus gradient
      boosted trees (the new best result, not the old JEPA pipeline) with
      gRodon and Phydon, with weights chosen from held out performance, not
      a plain average. Done 2026-09-17, see research_log.md and FINDINGS.md.
- [x] Test that blend using the same k-fold setup already used everywhere
      else in this project. Done as part of the item above.
- [x] If the blend wins, check it again across all seven seeds already
      established, the same way the earlier seed check was done, so it is
      not another result that only holds under one lucky seed. Done. Result
      is a real trade off, not a clean win: the blend never beats plain
      gradient boosted trees on overall R2, but it beats both individual
      models on Spearman in all seven seeds, and it substantially reduces
      the fast growth regime's worst failures in all seven seeds too, worst
      case R2 of -28.6 across all seven seeds against -109 and -627 for the
      two models alone. Which one to use depends on the goal, both numbers
      are kept in FINDINGS.md.

## Phase 2, know when to trust a prediction

- [x] Build a simple score for how far a new genome sits from anything in
      the training data. Done 2026-09-17, see `eval/confidence.py`.
- [ ] Add that score as a column on every prediction file this project
      produces, including the high quality genome report already sent.
      Not done. Holding off until the score itself actually works, see
      below, there is no point shipping a column that does not mean
      anything.
- [x] Confirm it actually catches the known problem case, the high GC, small
      genome cluster that got mispredicted as extreme slow growers. Done,
      and it does not work. See research_log.md for the full story. First
      test looked like a win, ten known problem genomes all got flagged.
      But checking the flag rate across the whole 7,873-genome deployment
      set, every single genome got flagged, 100 percent. The ten known
      problem genomes are not even unusual within that set, nine of them
      sit below the median distance. Likely cause is distance concentration
      in high dimensions, a known problem with plain nearest-neighbor
      distance once there are many features and a small training set. This
      approach does not work as tested. A real fix would need a different
      method, not just a different threshold, for example a much lower
      dimensional reduction first, checking each branch separately instead
      of one combined vector, or a density-based approach. Not attempted
      yet, left open for later.

## Phase 3, fix the training data where it is actually thin

- [x] Write down every specific region of feature space where the labeled
      corpus is thin or skewed, starting from Phase 0's findings. Done
      2026-09-17. Found ten thin or skewed regions with a finer grid scan,
      not just the two spotted by eye. The two that actually matter for
      real deployment genomes: the known thermophile-heavy region (1,580
      real GEM genomes, 3.0 percent of the whole catalog) and a newly found
      Vibrio-heavy region (918 real genomes, 1.7 percent), which pushes
      predictions too fast the way the thermophile region pushes them too
      slow. Confirmed the Vibrio-region bias directly against real
      predictions, median 0.46 hours there versus 0.60 hours elsewhere.
- [x] Look for any real, published growth rate measurements for ordinary
      organisms in those specific gaps, not just any new species at random.
      Done. Found 92 real, labeled species missing from the corpus only
      because they lacked a genus-level tree match. Built a family-level
      fallback (`taxonomic_centroid_embeddings`) and pulled real data for
      the 39 that were addable even with that fallback. Checked directly:
      none of the 39 land in either known gap region. Real, honest
      negative result, not the fix hoped for.
- [x] If more genus level approximate species get added, only add the ones
      that actually fill a known gap, not every available one regardless of
      where it lands. Followed this rule by checking first, found none of
      the 39 addable species qualified, and confirmed empirically across
      four seeds that adding them anyway made the benchmark flat or worse,
      never better. The right next step is not adding more species from
      what is currently reachable, none of it lands where the problem
      actually is, it is finding real data specifically in the thermophile
      or Vibrio regions, which needs a source beyond gRodon and Madin.

## Phase 4, grow the real labeled corpus

- [x] Ask about any additional real growth rate data beyond gRodon and
      Madin, especially for the kinds of organisms Phase 3 says are missing.
      Done 2026-09-17, see research_log.md and FINDINGS.md. Checked BacDive
      (no growth-rate field at all), EGGO (predictions from gRodon itself,
      not real measurements, ruled out), and a bioRxiv preprint (could not
      verify, site blocked every access attempt). Found the real answer
      already sitting locally: Madin et al. 2020's full dataset has 502
      species with real doubling time, genome size, and GC already
      computed, more than the ~389 this project's pipeline normally pulls.
      Found 11 of those landing exactly in the two known gaps, pulled real
      data for the 10 that resolved to a real accession, all 10 turned out
      to be exact GTDB tree tips. Benchmarked across four seeds against
      both existing corpora: Spearman worse in all four seeds against both,
      R2 mixed. Even this carefully targeted addition did not clearly help.
      Not adopted as the default corpus, following the user's own rule (use
      it only if it helps), but kept and documented as a real result.
- [x] Track down the lab added species file mentioned early in this project
      that was never actually delivered, and use it properly if it exists.
      Checked, per the user, via an online search rather than assuming it
      does not exist. Nothing found under that description; the real,
      usable find from this search was the fuller Madin et al. dataset
      above, which serves the same purpose this file was meant to.
- [x] Reconsider whether requiring an exact GTDB tree tip is still the right
      rule, now that there is real evidence about how the approximate
      placement method affects results. Done 2026-09-17, see
      research_log.md and FINDINGS.md. Ran the clean three-way comparison
      that had never actually been done, 175 exact-tip-only species against
      304 with genus approximation against 343 with family approximation
      added on top, same four seeds, same benchmark. Genus-level
      approximation is a clean win, better R2 and Spearman in all four
      seeds, no exceptions, this reverses an earlier finding that was true
      for the old model but not for the one that matters now. Family-level
      approximation is a net negative in three of four seeds. Updated
      conclusion: exact-tip-only should not be the default anymore, genus
      relaxation should be, family relaxation should not be used without a
      better reason than availability.

## Phase 5, re-check everything before trusting it

- [ ] Any change from Phases 1 through 4 gets retrained and re-evaluated the
      same full way as everything else in this project: both checkpoints,
      the benchmark, the necessity and sufficiency check, and more than one
      seed.
- [ ] Update FINDINGS.md with whatever actually happened, including if
      something here does not work out.

## Notes

Phases 0 and 1 are the ones worth doing first. They are cheap and could show
a real, honest improvement quickly. Phase 2 does not raise accuracy by
itself but makes every prediction this project produces more trustworthy,
worth doing alongside the first two. Phases 3 and 4 are the ones most likely
to actually move the accuracy number, but they depend on finding real data
this project does not currently have, so they take longer and are less
certain to pay off.
