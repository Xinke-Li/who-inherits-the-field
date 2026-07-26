#!/usr/bin/env python3
"""T2.4 - the checksum file for data/supplement/.

data/SHA256SUMS pins the ten frozen pre-registered tables and is never touched.
data/supplement/ carries what the revision produced beside them: the concept
event tables, the lineage tables, the early-concentration tables of T2.4, and
the reconstructed neuroscience funnel counters of T2.6. Those had no manifest.
This writes one, in data/SHA256SUMS's format, so the same `sha256sum -c` reads
both.

The B1 working cache at data/supplement/_ta_cache/ is excluded: it is an
in-progress fetch, not an artifact, and its shard count moves.

Usage:  python code/r51_supplement_sums.py [--check]
"""
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / "data" / "supplement"
SUMS = SUPP / "SHA256SUMS.supplement"
EXCLUDE_DIRS = {"_ta_cache"}


def entries():
    out = []
    for p in sorted(SUPP.rglob("*")):
        if not p.is_file() or p.name == SUMS.name:
            continue
        rel = p.relative_to(SUPP)
        if rel.parts[0] in EXCLUDE_DIRS:
            continue
        out.append((hashlib.sha256(p.read_bytes()).hexdigest(), rel.as_posix()))
    return out


def main(check):
    rows = entries()
    if check:
        want = {}
        for line in SUMS.read_text().splitlines():
            if line.strip():
                h, n = line.split(" *", 1)
                want[n] = h
        bad = [n for h, n in rows if want.get(n) != h]
        missing = [n for n in want if n not in {n for _, n in rows}]
        for n in bad:
            print(f"[FAIL] {n}")
        for n in missing:
            print(f"[MISSING] {n}")
        print(f"[r51] {len(rows) - len(bad)} of {len(rows)} verify; "
              f"{len(missing)} listed but absent")
        raise SystemExit(1 if (bad or missing) else 0)
    SUMS.write_text("".join(f"{h} *{n}\n" for h, n in rows))
    for h, n in rows:
        print(f"{h[:16]}… {n}")
    print(f"[r51] {len(rows)} files -> {SUMS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args().check)
