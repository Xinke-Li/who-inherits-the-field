#!/usr/bin/env python3
"""
Measure student-advisor co-authorship rate per AFT tree.
Run LOCALLY (this needs to scan all authorPub chunks; the cloud/device VM
times out at 45s per call, your own terminal has no such limit).

Usage:
    cd "C:\\Users\\lixin\\Desktop\\Economic Network"
    python measure_coauthorship.py

Outputs coauthorship_by_tree.json + prints a table.
Metric = fraction of relation=1 grad pairs whose student and advisor share
>=1 publication (authorPub score_total>=0.3). This is "ever co-authored",
an UPPER bound on the paper's early-window rate (econ 0.35 / neuro 0.76) but
preserves the cross-discipline ORDERING, which is what the de-confounding
argument needs. To get the early-window version, intersect shared pubids
with pub years from works_econ.jsonl / works_neuro.jsonl (see note at bottom).
"""
import csv, gzip, glob, json, random, os
from collections import defaultdict

SNAP = "aft_2026_snapshot"
TREES = ["econ", "physics", "chemistry", "math", "neuro"]
SAMPLE_PER_TREE = 8000      # set to None to use ALL pairs (slower)
SEED = 42
SCORE_MIN = 0.3

random.seed(SEED)
csv.field_size_limit(10**7)

# 1. pid -> primary tree
print("reading people ...")
prim = {}
with gzip.open(f"{SNAP}/people.csv.gz", "rt", encoding="utf-8", errors="replace") as f:
    r = csv.reader(f); h = next(r); im = h.index("majorarea"); ip = h.index("pid")
    for row in r:
        trees = [t.strip() for t in row[im].split(",") if t.strip()]
        if trees:
            prim[row[ip]] = trees[0]

# 2. grad pairs (relation=1) by trainee primary tree
print("reading connect ...")
pairs = defaultdict(list)
with gzip.open(f"{SNAP}/connect.csv.gz", "rt", encoding="utf-8", errors="replace") as f:
    r = csv.reader(f); h = next(r)
    i1 = h.index("pid1"); i2 = h.index("pid2"); ir = h.index("relation")
    for row in r:
        if row[ir] != "1":
            continue
        p1, p2 = row[i1], row[i2]
        if not p1 or not p2 or p2 == "0":
            continue
        t = prim.get(p1)
        if t in TREES:
            pairs[t].append((p1, p2))

sample = {}
needed = set()
for t in TREES:
    lst = pairs[t]
    random.shuffle(lst)
    s = lst if SAMPLE_PER_TREE is None else lst[:SAMPLE_PER_TREE]
    sample[t] = s
    for a, b in s:
        needed.add(a); needed.add(b)
print({t: len(sample[t]) for t in TREES}, "needed pids:", len(needed))

# 3. pid -> set(pubid) from authorPub (only needed pids, score>=0.3)
print("scanning authorPub chunks (this is the slow part) ...")
pubs = defaultdict(set)
for path in sorted(glob.glob(f"{SNAP}/authorPub*.csv.gz")):
    print("  ", os.path.basename(path))
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f); h = next(r)
        ipid = h.index("pid"); ipub = h.index("pubid"); isc = h.index("score_total")
        for row in r:
            pid = row[ipid]
            if pid in needed:
                try:
                    if float(row[isc]) >= SCORE_MIN:
                        pubs[pid].add(row[ipub])
                except ValueError:
                    pass

# 4. co-authorship rate per tree
out = {}
for t in TREES:
    n = len(sample[t]); co = 0
    for a, b in sample[t]:
        if pubs.get(a) and pubs.get(b) and (pubs[a] & pubs[b]):
            co += 1
    out[t] = {"pairs": n, "co_authored": co, "rate": round(co / n, 4) if n else None}
print(json.dumps(out, indent=2))
json.dump(out, open("coauthorship_by_tree.json", "w"), indent=2)
print("wrote coauthorship_by_tree.json")

# NOTE early-window refinement:
#   The paper's 0.35/0.76 is co-authorship WITHIN the first 5 years of the
#   student's career. authorPub has no year, so map shared pubids to years
#   via your works_*.jsonl (publication_year) or an OpenAlex batch lookup by
#   pmid/doi, then keep only shared pubs with year in [t0, t0+5]. The ordering
#   across trees will match the "ever" metric above; only the levels shift down.
