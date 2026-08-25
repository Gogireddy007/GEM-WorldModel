#!/usr/bin/env python
"""Streaming per-genome pass over the GEM MAG corpus: download each genome's
.fna.gz, compute real GC content (and, optionally, real 16S sequence
extraction via barrnap), then delete the downloaded file immediately, genomes are never accumulated on disk (52,515 genomes uncompressed would be
~130GB, far more than available disk space here).

Resumable: re-running skips genome_ids already present in the output CSV, so
a killed/interrupted run picks back up where it left off. Results are
flushed to disk every `--flush-every` genomes, not just at the end, so
partial progress is always inspectable and never lost.

GC-content is network/IO-bound (~1s/genome); barrnap is CPU-bound
(~7-8s/genome), pass --with-16s only when you actually want real 16S
sequences too (dominates total runtime at full 52,515-genome scale).
"""

import argparse
import concurrent.futures
import gzip
import random
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

from gem_worldmodel.features import genome_traits
from gem_worldmodel.features.rrna16s import kmer_profile
from gem_worldmodel.utils.config import load_config, resolve_path
from gem_worldmodel.utils.logging import get_logger

logger = get_logger(__name__)


def _get_with_retries(session: requests.Session, url: str, retries: int = 6, timeout: int = 45) -> requests.Response:
    """Connection/timeout errors were measured to spike heavily whenever two
    gem_slow_features.py jobs ran concurrently against the same host, one
    CPU-heavy (barrnap) job starves the other's I/O threads of scheduling,
    causing real read timeouts, not just flaky DNS. Retry with backoff+jitter
    absorbs transient failures, but the actual fix is not running two jobs
    against this host at once (see scripts/gem_slow_features.py's module
    docstring / README), this is a safety net, not a substitute for that.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(min(1.0 * (2**attempt), 20.0) + random.uniform(0, 1.0))
    raise last_exc


def process_one(genome_id: str, gem_cfg: dict, with_16s: bool, kmer_k: int, session: requests.Session) -> dict:
    url = f"{gem_cfg['base_url']}/{gem_cfg['genome_fna_path_template'].format(genome_id=genome_id)}"
    result = {"genome_id": genome_id, "gc_content": None, "genome_size_bp_verified": None}

    with tempfile.TemporaryDirectory() as tmp:
        gz_path = Path(tmp) / f"{genome_id}.fna.gz"
        fna_path = Path(tmp) / f"{genome_id}.fna"
        try:
            resp = _get_with_retries(session, url)
            gz_path.write_bytes(resp.content)
            with gzip.open(gz_path, "rb") as f_in, open(fna_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

            size, gc = genome_traits.genome_size_and_gc(fna_path)
            result["gc_content"] = gc
            result["genome_size_bp_verified"] = size

            if with_16s:
                rrna_counts = genome_traits.count_rrna_trna_genes(fna_path)
                result["rrna_16s_count_verified"] = rrna_counts["rrna_16s_count"]
                # attempted_16s marks that barrnap genuinely ran for this genome,
                # independent of whether it actually found a 16S copy, a real
                # zero-count result must still count as "done" for resumability,
                # not get silently retried forever (see resumability fix below).
                result["attempted_16s"] = True
                seq16s = _extract_first_16s_sequence(fna_path)
                if seq16s:
                    result["kmer_profile_16s"] = str(kmer_profile(seq16s, kmer_k))
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)

    return result


def _extract_first_16s_sequence(fna_path: Path) -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["barrnap", "--quiet", "--outseq", "/dev/stdout", str(fna_path)],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None

    seq_lines, capturing = [], False
    for line in out.splitlines():
        if line.startswith(">"):
            capturing = "16S" in line
            continue
        if capturing:
            seq_lines.append(line.strip())
    return "".join(seq_lines) if seq_lines else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-16s", action="store_true", help="also run barrnap for real 16S extraction (slow)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="process at most this many genomes (for testing)")
    parser.add_argument(
        "--output-name", type=str, default="unlabeled_corpus_features_enriched.csv",
        help="output filename under data/processed/, use a distinct name to run concurrently "
        "with another gem_slow_features.py invocation without both processes fighting over one file",
    )
    args = parser.parse_args()

    data_cfg = load_config("data")
    feat_cfg = load_config("features")
    processed_dir = resolve_path(data_cfg["paths"]["processed_dir"])
    gem_cfg = data_cfg["gem_portal"]
    kmer_k = feat_cfg["rrna16s"]["kmer_k"]

    base_path = processed_dir / "unlabeled_corpus_features.csv"
    out_path = processed_dir / args.output_name

    base = pd.read_csv(base_path)
    if out_path.exists():
        done = pd.read_csv(out_path)
        if args.with_16s and "attempted_16s" in done.columns:
            # "done" for a --with-16s run means barrnap actually ran for that
            # genome, NOT that it found a 16S copy. A genuine zero-copy result
            # is a valid, final outcome and must not be retried forever; only
            # genomes from a prior GC-only run (no attempted_16s marker at all)
            # still need (re)processing here.
            already = set(done.loc[done["attempted_16s"] == True, "genome_id"])  # noqa: E712
        elif args.with_16s:
            already = set()  # prior run never attempted 16S at all
        else:
            # A row with a real error (timeout, connection failure, etc.) is
            # NOT done, it must be retried, not silently treated as final.
            # Only rows that actually got a gc_content value count as done.
            has_error = done["error"].notna() if "error" in done.columns else pd.Series(False, index=done.index)
            already = set(done.loc[~has_error & done["gc_content"].notna(), "genome_id"])
        logger.info(f"resuming: {len(already)} genomes already processed (with_16s={args.with_16s})")
    else:
        done = pd.DataFrame(columns=["genome_id", "gc_content", "genome_size_bp_verified"])
        already = set()

    todo = base[~base["genome_id"].isin(already)]["genome_id"].tolist()
    if args.limit:
        # Shuffle before truncating, genome_ids are grouped by metagenome_id
        # in file order, so taking the first N would bias toward a handful of
        # source samples/ecosystems rather than a representative cross-section.
        rng = random.Random(0)
        rng.shuffle(todo)
        todo = todo[: args.limit]
    logger.info(f"{len(todo)} genomes remaining to process (with_16s={args.with_16s}, workers={args.workers})")

    # Drop any stale (e.g. GC-only) rows for genomes we're about to reprocess,
    # so the reprocessed row replaces rather than duplicates them.
    done = done[~done["genome_id"].isin(todo)].reset_index(drop=True)

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("https://", adapter)

    buffer = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, gid, gem_cfg, args.with_16s, kmer_k, session): gid for gid in todo
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            buffer.append(future.result())
            if i % args.flush_every == 0 or i == len(todo):
                done = pd.concat([done, pd.DataFrame(buffer)], ignore_index=True)
                done.to_csv(out_path, index=False)
                buffer = []
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta_min = (len(todo) - i) / rate / 60 if rate > 0 else float("nan")
                logger.info(f"{i}/{len(todo)} done ({rate:.2f}/s, ETA {eta_min:.1f} min)")

    logger.info(f"finished this run. Total rows in {out_path}: {len(done)}")


if __name__ == "__main__":
    main()
