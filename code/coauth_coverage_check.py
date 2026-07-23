#!/usr/bin/env python3
"""Confound check for the P1(a) coauthorship axis.

The "ever co-authored" rate = P(student and advisor share >=1 AFT authorPub
pubid, score>=0.3). This is mechanically sensitive to how MANY pubids each
person has linked: a tree whose persons carry more authorPub links will show a
higher shared-pubid rate even at equal true coauthorship. Before trusting the
cross-tree ORDERING we must see whether it is driven by differential linking
coverage rather than differential coauthorship.

Reports, per tree (same 8000-pair seeded sample as measure_coauthorship.py):
  * mean / median linked pubids per student and per advisor (score>=0.3)
  * frac of pairs where BOTH ends have >=1 linked pub ("both linkable")
  * raw rate            = P(share >=1 pubid)                       [all pairs]
  * coverage-cond rate  = P(share >=1 pubid | both ends linkable)  [controls coverage]
If the ORDERING survives on the coverage-conditioned rate, it is not a pure
coverage artifact.  Writes coauth_coverage_check.json.
"""
import csv, gzip, glob, json, random, os, statistics
from collections import defaultdict

SNAP = "aft_2026_snapshot"
TREES = ["econ", "physics", "chemistry", "math", "neuro"]
SAMPLE_PER_TREE = 8000
SEED = 42
SCORE_MIN = 0.3
random.seed(SEED)
csv.field_size_limit(10**7)

print("reading people ...", flush=True)
prim = {}
with gzip.open(f"{SNAP}/people.csv.gz", "rt", encoding="utf-8", errors="replace") as f:
    r = csv.reader(f); h = next(r); im = h.index("majorarea"); ip = h.index("pid")
    for row in r:
        toks = [t.strip() for t in row[im].split(",") if t.strip()]
        if toks:
            prim[row[ip]] = toks[0]

print("reading connect ...", flush=True)
pairs = defaultdict(list)
with gzip.open(f"{SNAP}/connect.csv.gz", "rt", encoding="utf-8", errors="replace") as f:
    r = csv.reader(f); h = next(r)
    i1 = h.index("pid1"); i2 = h.index("pid2"); ir = h.index("relation")
    for row in r:
        if row[ir] != "1":
            continue
        p1, p2 = row[i1], row[i2]
        if not p1 or not p2 or p2 == "0" or p1 == "0":
            continue
        if prim.get(p1) in TREES:
            pairs[prim[p1]].append((p1, p2))

sample, needed = {}, set()
for t in TREES:
    lst = pairs[t]; random.shuffle(lst)
    s = lst[:SAMPLE_PER_TREE]
    sample[t] = s
    for a, b in s:
        needed.add(a); needed.add(b)
print({t: len(sample[t]) for t in TREES}, "needed", len(needed), flush=True)

print("scanning authorPub ...", flush=True)
pubs = defaultdict(set)
for path in sorted(glob.glob(f"{SNAP}/authorPub*.csv.gz")):
    print("  ", os.path.basename(path), flush=True)
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

out = {}
for t in TREES:
    s = sample[t]; n = len(s)
    stu_links = [len(pubs.get(a, ())) for a, b in s]
    adv_links = [len(pubs.get(b, ())) for a, b in s]
    both_linkable = [(a, b) for a, b in s if pubs.get(a) and pubs.get(b)]
    co_all = sum(1 for a, b in s if pubs.get(a) and pubs.get(b) and (pubs[a] & pubs[b]))
    co_cond = sum(1 for a, b in both_linkable if pubs[a] & pubs[b])
    out[t] = {
        "pairs": n,
        "stu_links_mean": round(statistics.mean(stu_links), 2),
        "stu_links_median": statistics.median(stu_links),
        "adv_links_mean": round(statistics.mean(adv_links), 2),
        "adv_links_median": statistics.median(adv_links),
        "both_linkable": len(both_linkable),
        "both_linkable_frac": round(len(both_linkable) / n, 4),
        "rate_raw": round(co_all / n, 4),
        "rate_coverage_conditioned": round(co_cond / len(both_linkable), 4) if both_linkable else None,
    }
print(json.dumps(out, indent=2), flush=True)
json.dump(out, open("coauth_coverage_check.json", "w"), indent=2)
print("wrote coauth_coverage_check.json", flush=True)
