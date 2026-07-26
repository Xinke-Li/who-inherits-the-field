#!/usr/bin/env python3
"""T2.11, step 1: the per-author concept event tables the time contract is
verified against.

Section 7 currently says the shipped assertions cannot verify the time
contract, because verifying it needs per-work records and the release ships
only the aggregated columns. This emits those records, restricted to the
authors who actually appear in the five frozen modeling tables, so a third
party can rebuild every windowed feature from raw events and check that
nothing dated after t0+5 reaches one.

One row per (author, work, concept):

    author_id         bare OpenAlex author id
    publication_year  the work's year
    concept           display name, lower case
    score             the concept score OpenAlex assigned

The builder's view of a work is the first three concepts scoring at least 0.3
(build_neuro_dataset.CONCEPT_MIN_SCORE and its [:3] collapse); the cache stores
everything from 0.2 up. Only the builder's view is written here, so the table
is the one the frozen columns were actually computed from, not a superset that
would let a reader rebuild something the paper never used.

Records dated after t0+5 ARE present and must be: the label is read at t0+15.
The contract is that no such record enters a FEATURE, and that is what
reproduction/verify_time_contract.py asserts, exactly, per row.

The source is results/robustness/openalex_cache/, which r5_fetch_author_works
built from the live API in 2026, not the frozen 2026 snapshot the tables were
built from. Agreement is therefore high but not exact, and drift is expected at
the level Table 22 already reports. Only the t0+5 assertion is exact.

  python code/r36_concept_events.py
  python code/r36_concept_events.py --field econ

Output: data/supplement/concept_events_<field>.parquet
        data/supplement/concept_events_manifest.json
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "robustness" / "openalex_cache"
OUT = ROOT / "data" / "supplement"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
CONCEPT_MIN_SCORE, WORK_TOP_CONCEPTS = 0.3, 3
FLUSH_ROWS = 2_000_000

SCHEMA = pa.schema([
    ("author_id", pa.string()),
    ("publication_year", pa.int16()),
    ("concept", pa.string()),
    ("score", pa.float32()),
])


def bare(x):
    return x.rsplit("/", 1)[-1] if isinstance(x, str) else x


def wanted(fields):
    """author id -> the fields whose frozen table contains it, in either role."""
    who = {}
    per_field = {}
    for f in fields:
        d = pd.read_parquet(ROOT / "data" / f"clean_dataset_{f}.parquet",
                            columns=["st_openalex_id", "adv_openalex_id",
                                     "early_concepts"])
        d = d[d.early_concepts.apply(len) > 0]
        ids = set(d.st_openalex_id.dropna()) | set(d.adv_openalex_id.dropna())
        per_field[f] = ids
        for a in ids:
            who.setdefault(a, []).append(f)
    return who, per_field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default=None, choices=FIELDS)
    args = ap.parse_args()
    fields = [args.field] if args.field else FIELDS

    OUT.mkdir(parents=True, exist_ok=True)
    who, per_field = wanted(fields)
    print(f"[r36] {len(who)} distinct authors across {len(fields)} field(s)",
          flush=True)

    shards = sorted(glob.glob(str(CACHE / "*.parquet")))
    if not shards:
        raise SystemExit(f"no OpenAlex cache shards under {CACHE}")

    # r5's repair pass writes works_zrepair_*.parquet, which sorts after every
    # works_NNNNN shard and is meant to override the empty entries it replaces
    # (r6's loader lets later shards win). Taking the first occurrence would
    # keep the empty record and silently drop a repaired author, so resolve
    # which shard owns each author before emitting anything.
    owner = {}
    for i, shard in enumerate(shards):
        for a in pd.read_parquet(shard, columns=["aid"]).aid:
            if a in who:
                owner[a] = i
    print(f"[r36] {len(owner)} of them present in the cache", flush=True)

    writers, buf, counts = {}, {}, {}
    for f in fields:
        writers[f] = pq.ParquetWriter(OUT / f"concept_events_{f}.parquet",
                                      SCHEMA, compression="zstd")
        buf[f] = {"author_id": [], "publication_year": [], "concept": [],
                  "score": []}
        counts[f] = {"rows": 0, "works": 0, "authors": 0, "undated_works": 0}

    def flush(f, force=False):
        n = len(buf[f]["author_id"])
        if n and (force or n >= FLUSH_ROWS):
            writers[f].write_table(pa.Table.from_pydict(buf[f], schema=SCHEMA))
            buf[f] = {k: [] for k in buf[f]}

    for i, shard in enumerate(shards):
        d = pd.read_parquet(shard, columns=["aid", "works_json"])
        d = d[d.aid.map(lambda a: owner.get(a) == i)]
        for aid, wj in zip(d.aid, d.works_json):
            try:
                works = json.loads(wj)
            except Exception:
                continue
            fs = who[aid]
            rows, n_works, n_undated = [], 0, 0
            for w in works:
                y = w.get("year")
                if y is None:
                    n_undated += 1
                    continue
                n_works += 1
                for name, score in [c for c in (w.get("concepts") or [])
                                    if c[1] >= CONCEPT_MIN_SCORE
                                    ][:WORK_TOP_CONCEPTS]:
                    rows.append((int(y), name, float(score)))
            for f in fs:
                counts[f]["authors"] += 1
                counts[f]["works"] += n_works
                counts[f]["undated_works"] += n_undated
                counts[f]["rows"] += len(rows)
                b = buf[f]
                for y, name, score in rows:
                    b["author_id"].append(aid)
                    b["publication_year"].append(y)
                    b["concept"].append(name)
                    b["score"].append(score)
                flush(f)
        if (i + 1) % 20 == 0:
            print(f"[r36] shard {i+1}/{len(shards)}", flush=True)

    meta = {}
    for f in fields:
        flush(f, force=True)
        writers[f].close()
        p = OUT / f"concept_events_{f}.parquet"
        meta[f] = {
            "field": f,
            "authors_in_frozen_table": len(per_field[f]),
            "authors_with_cached_works": counts[f]["authors"],
            "author_coverage": round(counts[f]["authors"]
                                     / max(len(per_field[f]), 1), 4),
            "works": counts[f]["works"],
            "undated_works_dropped": counts[f]["undated_works"],
            "concept_rows": counts[f]["rows"],
            "megabytes": round(p.stat().st_size / 1e6, 1),
            "concept_min_score": CONCEPT_MIN_SCORE,
            "work_top_concepts": WORK_TOP_CONCEPTS,
            "source": "results/robustness/openalex_cache (r5_fetch_author_works)",
            "source_note": ("a live-API rebuild, not the frozen 2026 snapshot; "
                            "agreement with the frozen columns is high but not "
                            "exact and only the t0+5 assertion is exact"),
            "contains_post_t0plus5_records": True,
            "contains_note": ("by design: the label is read at t0+15, so the "
                              "late window must be present. The contract is "
                              "that no such record enters a feature."),
        }
        print(f"[{f:10}] authors {counts[f]['authors']:6d}/"
              f"{len(per_field[f]):6d}  works {counts[f]['works']:9d}  "
              f"concept rows {counts[f]['rows']:10d}  "
              f"{meta[f]['megabytes']:7.1f} MB", flush=True)

    mp = OUT / "concept_events_manifest.json"
    prev = json.loads(mp.read_text()) if mp.exists() else {}
    prev.update(meta)
    mp.write_text(json.dumps(prev, indent=2))
    print(f"\n-> {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
