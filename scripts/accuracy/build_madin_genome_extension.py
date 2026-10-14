#!/usr/bin/env python
"""This adds Madin species that have a doubling time and a genome accession but
were in neither the corpus nor madin_extension.csv, because Madin's trait table
has no genome size, GC or tRNA count for them.

Every trait is computed from the genome itself: the genomic FASTA is downloaded
from NCBI (the GenBank copy if the RefSeq record is suppressed), genome size and
GC are counted exactly, tRNA genes are counted with tRNAscan-SE using the
bacterial or archaeal model, and codon usage bias and IVYWREL come from the
same annotation-free gene pipeline used for GEM MAGs. tRNA counts in Madin's
table and in GEM's metadata are also tRNAscan-SE based; `--check-trna` compares
our counts with Madin's on genomes that have both.

Placement in GTDB is by the genome's own accession first (the exact genome),
then by species name, then by genus name. Doubling times are corrected to the
reference temperature as for the other rows.

Names like "Dictyoglomus sp." are skipped: Madin's value belongs to one
unnamed strain, and the accession table cannot say which.

With `--literature`, the same steps run on data/curated/literature_growth_rates.csv
(hand-checked values for lineages Madin's table lacks, each with the genome of
the measured strain) and write literature_extension.csv.

Writes madin_genome_extension.csv.
"""

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from gem_worldmodel.data import gtdb, ncbi_genomes
from gem_worldmodel.features import cub_pipeline, genome_traits, temperature
from gem_worldmodel.utils.config import load_config, resolve_path

HMM = "data/raw/ribosomal_pfam.hmm"
GENOMES = Path("data/raw/ncbi_genomes")


def clean_species(name):
    """Genus and species epithet, keeping a leading "Candidatus"."""
    words = str(name).replace("[", "").replace("]", "").split()
    return " ".join(words[:3] if words[:1] == ["Candidatus"] else words[:2])


def token(taxonomy, prefix):
    m = re.search(rf"{prefix}[^;]*", str(taxonomy))
    return m.group(0) if m and m.group(0) != prefix else None


def placer(taxonomy):
    by_acc = dict(zip(taxonomy["accession_bare"].str.replace(r"^GC[AF]_", "", regex=True).str.split(".").str[0],
                      taxonomy["gtdb_taxonomy"], strict=True))
    by_species, by_genus = {}, {}
    for t in taxonomy.drop_duplicates("gtdb_taxonomy")["gtdb_taxonomy"]:
        s, g = token(t, "s__"), token(t, "g__")
        if s:
            by_species.setdefault(s[3:], t)
        if g:
            by_genus.setdefault(re.sub(r"_[A-Z]+$", "", g[3:]), {}).setdefault(g, []).append(t)

    def place(species, accession):
        key = re.sub(r"^GC[AF]_", "", accession).split(".")[0]
        if key in by_acc:
            return by_acc[key], "accession"
        if species in by_species:
            return by_species[species], "species"
        groups = by_genus.get(species.split()[0])
        if groups:
            return re.sub(r"s__[^;]*", "s__", max(groups.values(), key=len)[0]), "genus"
        return None, None

    return place


def genome_path(accession):
    folder = GENOMES / accession
    path = cub_pipeline.genome_fasta(folder)
    if path is not None and path.suffix == ".gz":
        plain = path.with_suffix("")
        path = plain if plain.exists() else None
    return path or ncbi_genomes.fetch_genomic_fasta(accession, folder)


def work(job):
    species, accession, domain = job
    row = {"species": species, "accession": accession, "genome_size_bp": np.nan, "gc_content": np.nan,
           "trna_count": np.nan, "cub": np.nan, "ivywrel": np.nan, "n_ribosomal": np.nan}
    try:
        path = genome_path(accession)
        if path is None:
            print(f"{species} {accession}: no genome on NCBI", flush=True)
            return row
        row["genome_size_bp"], row["gc_content"] = genome_traits.genome_size_and_gc(path)
        row["trna_count"] = genome_traits.count_trna(path, domain, timeout=3600)
        g = cub_pipeline.genome_cub(path, HMM)
        row.update(cub=g["cub"], ivywrel=g["ivywrel"], n_ribosomal=g["n_ribosomal"])
    except Exception as exc:  # noqa: BLE001
        print(f"{species} {accession}: {str(exc)[:100]}", flush=True)
    return row


def run(jobs, workers):
    rows = []
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(work, jobs)):
            rows.append(row)
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(jobs)} genomes done", flush=True)
    return pd.DataFrame(rows)


def accession_table():
    acc = pyreadr.read_r("data/raw/Accession2Species_Madin.rda")["spp_acc"]
    acc.columns = ["accession", "name"]
    acc["species"] = acc["name"].map(clean_species)
    # prefer a RefSeq accession when a species has several
    acc = acc.sort_values("accession", key=lambda s: ~s.str.startswith("GCF_"), kind="stable")
    return acc.drop_duplicates("species").set_index("species")["accession"]


def check_trna(processed, workers):
    """Our tRNAscan-SE counts against Madin's on extension species with a cached genome."""
    ext = pd.read_csv(processed / "madin_extension.csv")
    first = accession_table()
    ext = ext[ext["species"].isin(first.index)].sample(n=24, random_state=0)
    jobs = [(s, first[s], "archaea" if t.startswith("d__Archaea") else "bacteria") for s, t in zip(ext.species, ext.gtdb_taxonomy, strict=True)]
    out = run(jobs, workers).merge(ext[["species", "trna_count", "genome_size_bp", "gc_content"]], on="species", suffixes=("", "_madin"))
    out = out.dropna(subset=["trna_count"])
    for col in ("trna_count", "genome_size_bp", "gc_content"):
        a, b = out[col], out[f"{col}_madin"]
        print(f"{col}: n={len(out)}, Pearson {np.corrcoef(a, b)[0, 1]:.3f}, median ratio ours/Madin {np.median(a / b):.3f}")
    print(out[["species", "trna_count", "trna_count_madin"]].to_string(index=False))


def build(new, data_cfg, feat_cfg, out_path, workers, extra=()):
    """Place, compute traits from the genome, add the temperature target and write."""
    taxonomy = pd.concat([gtdb.fetch_bac_taxonomy(data_cfg), gtdb.fetch_arc_taxonomy(data_cfg)], ignore_index=True)
    place = placer(taxonomy)
    placed = [place(s, a) for s, a in zip(new["species"], new["accession"], strict=True)]
    new = new.assign(gtdb_taxonomy=[p[0] for p in placed], match_rank=[p[1] for p in placed])
    print("placement:", new["match_rank"].value_counts(dropna=False).to_dict())
    new = new.dropna(subset=["gtdb_taxonomy"])

    jobs = [(s, a, "archaea" if t.startswith("d__Archaea") else "bacteria")
            for s, a, t in zip(new["species"], new["accession"], new["gtdb_taxonomy"], strict=True)]
    # traits already computed for an accession are reused (tRNAscan-SE is the slow step)
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["accession"])
    trait_cols = ["genome_size_bp", "gc_content", "trna_count", "cub", "ivywrel", "n_ribosomal"]
    cached = done.drop_duplicates("accession").set_index("accession")
    todo = [j for j in jobs if j[1] not in cached.index]
    print(f"{len(jobs) - len(todo)} genomes reused from {out_path.name}, {len(todo)} to compute")
    traits = run(todo, workers) if todo else pd.DataFrame(columns=["species", "accession", *trait_cols])
    reused = new[new["accession"].isin(cached.index)][["species", "accession"]]
    reused = reused.join(cached[trait_cols], on="accession")
    traits = pd.concat([traits, reused], ignore_index=True)
    df = new.merge(traits, on=["species", "accession"])
    df = df.rename(columns={"d": "doubling_time_hours", "GrowthTemp": "growth_temp_c"})
    df = df.dropna(subset=["genome_size_bp", "trna_count"])
    df = temperature.add_reference_temperature_target(df, feat_cfg)
    cols = ["species", "accession", "doubling_time_hours", "growth_temp_c", "doubling_time_hours_ref", "temp_corrected",
            "genome_size_bp", "gc_content", "trna_count", "cub", "ivywrel", "n_ribosomal", "gtdb_taxonomy", "match_rank", *extra]
    out = df[cols].sort_values("species").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    arch = out["gtdb_taxonomy"].str.startswith("d__Archaea")
    print(f"wrote {out_path.name}: {len(out)} species ({arch.sum()} archaeal), "
          f"{(out['doubling_time_hours'] > 24).sum()} slower than 24 h, {out['cub'].notna().sum()} with gene features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-trna", action="store_true")
    ap.add_argument("--literature", action="store_true")
    args = ap.parse_args()

    data_cfg, feat_cfg = load_config("data"), load_config("features")
    processed = resolve_path(data_cfg["paths"]["processed_dir"])
    if args.check_trna:
        check_trna(processed, args.workers)
        return

    if args.literature:
        lit = pd.read_csv("data/curated/literature_growth_rates.csv")
        lit = lit.rename(columns={"doubling_time_hours": "d", "growth_temp_c": "GrowthTemp"})
        print(f"{len(lit)} literature rows, {(lit['d'] > 24).sum()} slower than 24 h")
        build(lit, data_cfg, feat_cfg, processed / "literature_extension.csv", args.workers, extra=["strain", "source", "doi"])
        return

    growth = pyreadr.read_r("data/raw/GrowthRates_Madin.rda")["d"]
    growth["species"] = growth["species"].map(clean_species)
    growth = growth[~growth["species"].str.endswith(" sp.")]
    growth = growth.drop_duplicates("species")
    used = set(pd.read_csv(processed / "features_sample_expanded.csv")["species"]) | set(pd.read_csv(processed / "madin_extension.csv")["species"])
    first = accession_table()
    new = growth[~growth["species"].isin(used) & growth["species"].isin(first.index)].copy()
    new["accession"] = new["species"].map(first)
    print(f"{len(new)} unused Madin species with a genome accession, {(new['d'] > 24).sum()} slower than 24 h")

    build(new, data_cfg, feat_cfg, processed / "madin_genome_extension.csv", args.workers)

if __name__ == "__main__":
    main()
