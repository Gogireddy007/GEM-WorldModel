"""Gene-content features for energy metabolism and lifestyle, from a genome alone.

Genes are predicted with the same gene finder as the codon usage pipeline and
searched against a small set of Pfam families with Pfam's curated gathering
thresholds. The families mark ways of making a living that are linked to slow
growth (chemolithotrophy, methanogenesis, sulfate reduction, nitrogen fixation,
carbon fixation) or to fast, copiotrophic growth (many transporters and
regulators, motility, aerobic respiration). Each family is named by its Pfam
accession so the list survives Pfam renaming.

`genome_lifestyle` returns the number of proteins hitting each family plus the
gene count. `lifestyle_features` turns those counts into the model inputs.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from gem_worldmodel.features import cub_pipeline

FAMILIES = {
    "PF00115": "heme-copper oxidase subunit I",
    "PF01654": "cytochrome bd oxidase subunit I",
    "PF00361": "complex I membrane subunit",
    "PF00374": "NiFe hydrogenase large subunit",
    "PF02906": "FeFe hydrogenase",
    "PF02249": "methyl-coenzyme M reductase alpha",
    "PF02745": "methyl-coenzyme M reductase alpha, N-terminal",
    "PF02241": "methyl-coenzyme M reductase beta",
    "PF08679": "dissimilatory sulfite reductase DsrD",
    "PF00142": "nitrogenase iron protein NifH",
    "PF00148": "nitrogenase MoFe protein NifD/K",
    "PF00016": "RuBisCO large subunit",
    "PF02461": "ammonia/methane monooxygenase subunit A",
    "PF00124": "photosynthetic reaction centre",
    "PF00223": "photosystem I PsaA/PsaB",
    "PF00005": "ABC transporter ATPase",
    "PF00528": "ABC transporter permease",
    "PF07690": "major facilitator superfamily",
    "PF00083": "sugar transporter",
    "PF02378": "PTS sugar transporter EIIC",
    "PF06808": "TRAP transporter large permease",
    "PF00593": "TonB-dependent receptor",
    "PF00072": "response regulator receiver",
    "PF00512": "histidine kinase dimerisation",
    "PF04542": "sigma-70 region 2",
    "PF03466": "LysR substrate binding",
    "PF00440": "TetR family regulator",
    "PF00392": "GntR family regulator",
    "PF12833": "AraC family regulator",
    "PF01047": "MarR family regulator",
    "PF00196": "LuxR family regulator",
    "PF00669": "flagellin N-terminal",
    "PF00015": "methyl-accepting chemotaxis protein",
    "PF00199": "catalase",
    # energy metabolism, added 2026-10-15 (energy_features)
    "PF14710": "respiratory nitrate reductase NarG, N-terminal",
    "PF02665": "respiratory nitrate reductase gamma subunit NarI",
    "PF03892": "periplasmic nitrate reductase NapB",
    "PF02239": "cd1 nitrite reductase NirS",
    "PF18764": "nitrous oxide reductase NosZ",
    "PF04976": "DMSO/TMAO reductase anchor DmsC",
    "PF02300": "fumarate reductase subunit C",
    "PF02313": "fumarate reductase subunit D",
    "PF11854": "outer membrane porin MtrB/PioB",
    "PF22111": "decaheme cytochrome MtrC/MtrF",
    "PF22112": "decaheme cytochrome OmcA",
    "PF03598": "CO dehydrogenase/acetyl-CoA synthase beta",
    "PF03599": "CO dehydrogenase/acetyl-CoA synthase delta",
    "PF00213": "F-type ATP synthase delta (OSCP)",
    "PF00119": "F-type ATP synthase subunit a",
    "PF01496": "V/A-type ATP synthase subunit I",
    "PF01991": "V/A-type ATP synthase subunit E",
    "PF05896": "Na+-translocating NADH:quinone oxidoreductase NqrA",
    "PF24836": "NqrA second domain (not in RnfC)",
    "PF04400": "Na+-NQR maturation protein NqrM",
    "PF13375": "Rnf complex subunit RnfC",
    "PF03977": "Na+-pumping decarboxylase beta subunit",
    "PF04208": "Na+-pumping methyltransferase MtrA",
    "PF02901": "pyruvate formate-lyase",
    "PF00871": "acetate kinase",
    "PF01515": "phosphotransacetylase",
}

# searched with an E-value cutoff instead of the gathering threshold: the
# archaeal AmoC of ammonia-oxidizing archaea scores below Pfam's curated cutoff
# (bit score about 50) but is found at E about 1e-14
EVALUE_FAMILIES = {"PF04896": "ammonia/methane monooxygenase subunit C"}
EVALUE = 1e-5

TRANSPORTERS = ["PF00005", "PF00528", "PF07690", "PF00083", "PF02378", "PF06808", "PF00593"]
REGULATORS = ["PF00072", "PF00512", "PF04542", "PF03466", "PF00440", "PF00392", "PF12833", "PF01047", "PF00196"]

FEATURES = [
    "aerobic_respiration", "oxidase_count", "complex_i", "hydrogenase_count", "methanogenesis",
    "sulfate_reduction", "nitrogen_fixation", "rubisco", "ammonia_monooxygenase", "phototrophy",
    "transporters_per_1000", "regulators_per_1000", "motility", "catalase",
]


ENERGY_FEATURES = [
    "nitrate_respiration", "denitrification", "dmso_tmao_reduction", "fumarate_reductase", "extracellular_electron_transfer",
    "wood_ljungdahl", "electron_acceptor_classes", "f_type_atpase", "va_type_atpase", "na_nqr", "rnf",
    "na_decarboxylase", "sodium_pumps", "fermentation_genes", "fermentation_only",
]


def family_hits(proteins: list[str], hmm_path, evalue_hmm_path=None, workdir=None) -> dict[str, int]:
    """Number of proteins hitting each Pfam accession.

    Families in hmm_path use the gathering threshold, families in
    evalue_hmm_path (EVALUE_FAMILIES) use the E-value cutoff EVALUE.
    """
    seen = {acc: set() for acc in [*FAMILIES, *EVALUE_FAMILIES]}
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        faa = Path(tmp) / "p.faa"
        with open(faa, "w") as handle:
            for i, p in enumerate(proteins):
                if p:
                    handle.write(f">{i}\n{p}\n")
        searches = [(hmm_path, ["--cut_ga"])]
        if evalue_hmm_path is not None:
            searches.append((evalue_hmm_path, ["-E", str(EVALUE)]))
        for n, (hmm, cutoff) in enumerate(searches):
            tbl = Path(tmp) / f"hits{n}.tbl"
            subprocess.run(
                ["hmmsearch", *cutoff, "--noali", "--cpu", "1", "--tblout", str(tbl), "-o", "/dev/null", str(hmm), str(faa)],
                check=True,
            )
            for line in open(tbl):
                if line.startswith("#"):
                    continue
                fields = line.split()
                acc = fields[3].split(".")[0]
                if acc in seen:
                    seen[acc].add(fields[0])
    return {acc: len(v) for acc, v in seen.items()}


def genome_lifestyle(genome_path, hmm_path, evalue_hmm_path=None, workdir=None) -> dict:
    """Pfam family counts and gene count for one genome FASTA (plain or gzipped)."""
    _, proteins = cub_pipeline.predict_genes(cub_pipeline.read_contigs(genome_path))
    return {**family_hits(proteins, hmm_path, evalue_hmm_path, workdir), "n_genes": len(proteins)}


def lifestyle_features(counts) -> np.ndarray:
    """Model inputs from a table (DataFrame) of family counts and n_genes, in FEATURES order."""
    c = counts
    per_1000 = 1000.0 / c["n_genes"].to_numpy(float)
    phototroph = (c["PF00124"] + c["PF00223"]) > 0
    # chlorophyll synthesis shares the NifD/K family: cyanobacteria (both reaction
    # centre types) carry two members (ChlN, ChlB), other phototrophs four
    # (BchN, BchB, BchY, BchZ), so only members beyond those count toward nitrogenase
    cyanobacterium = (c["PF00124"] > 0) & (c["PF00223"] > 0)
    pigment = np.where(cyanobacterium, 2, np.where(phototroph, 4, 0))
    nif_dk = c["PF00148"] - pigment
    cols = [
        ((c["PF00115"] + c["PF01654"]) > 0),
        c["PF00115"] + c["PF01654"],
        c["PF00361"] > 0,
        c["PF00374"] + c["PF02906"],
        (c["PF02249"] + c["PF02745"] + c["PF02241"]) > 0,
        c["PF08679"] > 0,
        (c["PF00142"] > 0) & (nif_dk >= 2),
        c["PF00016"] > 0,
        (c["PF02461"] + c["PF04896"]) > 0,
        phototroph,
        c[TRANSPORTERS].sum(axis=1) * per_1000,
        c[REGULATORS].sum(axis=1) * per_1000,
        c["PF00669"] > 0,
        c["PF00199"] > 0,
    ]
    out = np.column_stack([np.asarray(col, dtype=float) for col in cols])
    out[c["n_genes"].isna().to_numpy()] = np.nan
    return out


def energy_features(counts) -> np.ndarray:
    """Energy-metabolism inputs from a table of family counts, in ENERGY_FEATURES order.

    Electron acceptor classes counted: oxygen (terminal oxidase), nitrate (NarG or NapB), nitrous oxide
    (NosZ, the denitrification marker), sulfate (DsrD), metals outside the cell (Mtr/Omc), DMSO or TMAO, fumarate
    (E. coli-type reductase only) and CO2 (methyl-coenzyme M reductase or the Wood-Ljungdahl
    CO dehydrogenase/acetyl-CoA synthase). "Fermentation only" means fermentation genes and none of
    those acceptors and no phototrophy, so an organism using an acceptor these families miss is
    counted as a fermenter. Rows without gene counts come back as NaN.

    Checked against known organisms (2026-10-15): the NirS heme d1 family (PF02239) and the NarI
    gamma subunit (PF02665) are counted but not used, since they hit many organisms that neither
    denitrify nor respire nitrate (a common propeller fold and a generic membrane cytochrome); Na-NQR
    needs an NqrA-specific domain or NqrM besides the NqrA barrel, which E. coli's RnfC-like protein hits;
    the Wood-Ljungdahl call uses the acetyl-CoA synthase beta subunit (PF03598) only, since the delta
    subunit family (PF03599) also hits Bacillus, Anabaena and Caldicellulosiruptor. Known misses: the NarG
    of Mycobacterium tuberculosis lacks the N-terminal domain used for nitrate respiration, fumarate
    reductases other than the E. coli type are not detected, and syntrophy has no single-gene marker
    (the four syntrophs in the labeled set share no pattern that separates them from other anaerobes).
    """
    c = counts

    def has(*accs):
        return c[list(accs)].sum(axis=1) > 0

    oxygen = has("PF00115", "PF01654")
    nitrate = has("PF14710", "PF03892")
    denitrification = has("PF18764")
    sulfate = has("PF08679")
    eet = has("PF11854", "PF22111", "PF22112")
    dmso = has("PF04976")
    fumarate = has("PF02300", "PF02313")
    methanogen = has("PF02249", "PF02745", "PF02241")
    wood_ljungdahl = has("PF03598")
    acceptors = (oxygen.astype(int) + nitrate + denitrification + sulfate + eet + dmso + fumarate + (methanogen | wood_ljungdahl))
    phototroph = has("PF00124", "PF00223")
    na_nqr = has("PF05896") & has("PF24836", "PF04400")
    rnf, na_dec, mtr = has("PF13375"), has("PF03977"), has("PF04208")
    fermentation = has("PF02901") | (has("PF00871") & has("PF01515"))
    cols = [
        nitrate, denitrification, dmso, fumarate, eet, wood_ljungdahl, acceptors,
        has("PF00213", "PF00119"), has("PF01496", "PF01991"), na_nqr, rnf, na_dec,
        na_nqr.astype(int) + rnf + na_dec + mtr,
        fermentation, fermentation & (acceptors == 0) & ~phototroph,
    ]
    out = np.column_stack([np.asarray(col, dtype=float) for col in cols])
    out[c["n_genes"].isna().to_numpy()] = np.nan
    return out
