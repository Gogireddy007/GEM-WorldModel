# Curated growth rates from the literature

`literature_growth_rates.csv` holds measured doubling times for lineages with
no rate in Madin et al.'s table (ammonia-oxidizing archaea, Nitrospira and
other nitrite oxidizers, Verrucomicrobia). Each row was checked against the
primary paper or its table, is a pure culture of the strain named, and is
paired with the genome of that same strain. Where a paper gives a specific
growth rate mu instead of a doubling time, the doubling time is ln(2)/mu.

Rows were left out when the value could not be confirmed in the primary
source, when it came from an enrichment culture, or when it described
something other than population growth (a budding cycle, time to colonies).
Examples left out: anammox (enrichment cultures), Gemmata obscuriglobus (the
often quoted 13 h is a budding cycle), comammox Nitrospira inopinata (no
growth rate reported), Ca. Nitrotoga arctica (enrichment).

`label_exclusions.csv` lists growth rates in Madin et al.'s table that are
almost certainly not maximal growth at the stated temperature, each with the
evidence and source. They are dropped from every model and test by
`deploy_features.load_training_table`. Four methanotroph values come from one
study of methanotroph growth at low temperatures (Kevbrina et al. 2001) by way
of Corkrey et al.'s compilation, where they are 2,700 to 4,600 h at 20 to 30 C
(Madin's values are about a 24th of that); the same study gives Methylococcus
capsulatus about 4,600 h against 1.87 h elsewhere. The other five were found by
`scripts/accuracy/screen_labels.py`, which flags species at least 5 times away
from both their held-out prediction and their genus relatives, and were kept
out only where a second source or the study itself showed the value is not a
maximal rate. Most flagged species (Mycobacterium leprae, psychrotolerant
methanogens, Clostridium perfringens) are real and stay in. The screen cannot
catch an error shared by a whole genus, which is how the methanotrophs slipped
through it.

Values describe the conditions in the paper, which are not always the
organism's optimum, the same caveat Madin's compilation carries. For example
Nowka et al. list Nitrobacter winogradskyi at 26 h while Madin's table has
16 h.
