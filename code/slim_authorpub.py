#!/usr/bin/env python3
"""One-pass slim extract of the AFT authorPub anchor columns for the whole suite.

resolve_by_pubid.build_pub_index re-scans the full 1.6 GB authorPub on every
fetch AND every apply. Across math/physics/chemistry(/econ), fetch+apply,
that is ~8 full scans. Instead we scan ONCE here, keep only the rows we will ever
need -- pid in the union of all suite persons, score_total >= 0.3 -- and the four
anchor columns, and write a single small parquet. Every resolution then points
--pubs at this file and reads it in a second.

This changes NO resolution semantics: build_pub_index applies score>=MIN_SCORE
(0.3) and picks anchors per pid exactly as before; we only pre-drop rows it would
have discarded anyway. Persons absent here simply have no anchor pubs (same as a
full scan finding none).

Usage:
  python slim_authorpub.py --snap aft_2026_snapshot \
      --detailed suite_data/math/Math_Pairs_Detailed.parquet \
                 suite_data/physics/Physics_Pairs_Detailed.parquet \
                 suite_data/chemistry/Chemistry_Pairs_Detailed.parquet \
      --out suite_data/aft_authorpub_slim.parquet
"""
import argparse, csv, gzip, glob, os
import pandas as pd

csv.field_size_limit(10**7)
SCORE_MIN = 0.3


def gather_pids(detailed_paths):
    pids = set()
    for p in detailed_paths:
        df = pd.read_parquet(p, columns=["advisor_pid", "student_pid"])
        pids.update(df.advisor_pid.astype(str))
        pids.update(df.student_pid.astype(str))
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", default="aft_2026_snapshot")
    ap.add_argument("--detailed", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pids = gather_pids(args.detailed)
    print(f"[slim] {len(pids):,} unique suite persons to keep", flush=True)

    rows_pid, rows_pmid, rows_doi, rows_sc = [], [], [], []
    kept = 0
    for path in sorted(glob.glob(os.path.join(args.snap, "authorPub*.csv.gz"))):
        print(f"[slim] scanning {os.path.basename(path)} (kept so far {kept:,})", flush=True)
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            r = csv.reader(f); h = next(r)
            ip = h.index("pid"); ipm = h.index("pmid"); idoi = h.index("doi"); isc = h.index("score_total")
            for row in r:
                pid = row[ip]
                if pid not in pids:
                    continue
                try:
                    sc = float(row[isc])
                except ValueError:
                    continue
                if sc < SCORE_MIN:
                    continue
                rows_pid.append(pid); rows_pmid.append(row[ipm])
                rows_doi.append(row[idoi]); rows_sc.append(sc)
                kept += 1
    out = pd.DataFrame({"pid": rows_pid, "pmid": rows_pmid, "doi": rows_doi, "score_total": rows_sc})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[slim] wrote {args.out}: {len(out):,} rows, "
          f"{out.pid.nunique():,} persons with >=1 anchor pub (score>=0.3)", flush=True)


if __name__ == "__main__":
    main()
