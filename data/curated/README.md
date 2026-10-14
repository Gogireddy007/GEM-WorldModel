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

Values describe the conditions in the paper, which are not always the
organism's optimum, the same caveat Madin's compilation carries. For example
Nowka et al. list Nitrobacter winogradskyi at 26 h while Madin's table has
16 h.
