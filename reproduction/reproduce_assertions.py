#!/usr/bin/env python3
"""Assertion-style reproduction. Recomputes the headline numbers from the frozen
tables in ../data and checks them against the recorded SHA-256, base rates, and
label definition. Prints one line per check and exits non-zero on any failure.

Run from the package root:  python reproduction/reproduce_assertions.py
"""
import hashlib, json, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
THETA = 0.2
fails = []


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


# 1. SHA-256 of every table matches SHA256SUMS
sums = {}
for line in open(os.path.join(DATA, "SHA256SUMS")):
    h, fn = line.split()
    sums[fn.lstrip("*").strip()] = h.strip()
for f in FIELDS:
    fn = f"clean_dataset_{f}.parquet"
    path = os.path.join(DATA, fn)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    check(f"{f}: SHA-256 matches SHA256SUMS", got == sums.get(fn))

# 2. base rate and n_students match the summary; label equals (late_overlap > theta)
for f in FIELDS:
    df = pd.read_parquet(os.path.join(DATA, f"clean_dataset_{f}.parquet"))
    s = json.load(open(os.path.join(DATA, f"dataset_summary_{f}.json")))
    check(f"{f}: n_students matches summary ({len(df)})", len(df) == s["n_students"])
    check(f"{f}: base rate matches summary ({round(df.y.mean(),4)})",
          abs(df.y.mean() - s["label_base_rate"]) < 5e-4)
    check(f"{f}: label equals 1[late_overlap>{THETA}]",
          ((df.late_overlap > THETA).astype(int) == df.y).all())
    check(f"{f}: no right-censored rows (t0+15<=2026)", (df.t0 + 15 <= 2026).all())

# 3. co-authorship axis ordering: econ is the lowest early-window rate
axis = json.load(open(os.path.join(DATA, "coauthorship_axis.json")))["axis"]
rates = {k: v["early_window_coauth_openalex"] for k, v in axis.items()}
check("axis: econ has the lowest early-window coauth",
      rates["econ"] == min(rates.values()))

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
print("all reproduction checks passed")
