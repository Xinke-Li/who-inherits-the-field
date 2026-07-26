#!/usr/bin/env python3
"""T2.10(b): how much of the within-cohort scramble residual can composition explain?

e9a's cohort variant permutes y within (split, t0) cells, which preserves each
cell's base rate. Chemistry is the only discipline whose residual departs from
chance (AUC-ROC 0.5247, t 3.205, p 0.003), and the paper calls that residual
"compositional" without measuring it.

This measures it. The composition-only model sees no features at all: every test
row is scored by the base rate of its own (split, t0) cell. That is the most a
model can extract from cell composition, so its AUC-ROC is the upper bound on
what composition can explain. It is evaluated both against the real label and
against within-cohort scrambled labels at the same 30 seeds e9a uses.

  python code/r22_composition_bound.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_10b_composition_bound"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
SEEDS = list(range(30))


def run_one(field):
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
    from sklearn.metrics import roc_auc_score
    from utils import data as D

    dsp = D.temporal_split(D.load_dataset())
    te = dsp[dsp.split == "test"]
    y = te.y.values.astype(int)
    t0 = te.t0.values

    # composition-only score: the base rate of the row's own (split, t0) cell.
    # Within the test split, split is constant, so the cell is t0.
    rate = {c: float(y[t0 == c].mean()) for c in np.unique(t0)}
    score = np.array([rate[c] for c in t0])

    real = float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan")

    # the same score against within-cohort scrambled labels, e9a's protocol
    scr = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ys = y.copy()
        for c in np.unique(t0):
            idx = np.where(t0 == c)[0]
            ys[idx] = rng.permutation(y[idx])
        if len(set(ys)) > 1:
            scr.append(float(roc_auc_score(ys, score)))

    out = {
        "field": field,
        "n_test": int(len(y)),
        "n_cells": int(len(rate)),
        "test_base_rate": round(float(y.mean()), 4),
        "composition_only_auc_roc_vs_real_label": round(real, 4),
        "composition_only_auc_roc_vs_scrambled": {
            "mean": round(float(np.mean(scr)), 4),
            "std": round(float(np.std(scr)), 4),
            "n_seeds": len(scr),
        },
        "note": ("the model uses no features; each test row is scored by the base "
                 "rate of its own (split, t0) cell, which is the maximum "
                 "information cell composition carries"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{field}.json", "w"), indent=2)
    print(f"[T2.10b] {field:10} n_test {out['n_test']:>5} cells {out['n_cells']:>3}  "
          f"composition-only AUC-ROC vs real {real:.4f}  "
          f"vs scrambled {out['composition_only_auc_roc_vs_scrambled']['mean']:.4f}")
    return 0


def main():
    if os.environ.get("R22_FIELD"):
        return run_one(os.environ["R22_FIELD"])
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        env = dict(os.environ)
        env.update(R22_FIELD=f, DATASET=f,
                   DATASET_PATH=str(ROOT / "data" / f"clean_dataset_{f}.parquet"))
        env.pop("NEURO_DATASET", None)
        subprocess.call([sys.executable, str(Path(__file__).resolve())], env=env)

    per = {f: json.load(open(OUT / f"{f}.json")) for f in FIELDS
           if (OUT / f"{f}.json").exists()}
    obs = {}
    for f in FIELDS:
        p = ROOT / "results" / f"results_{f}" / "e9a_placebo.json"
        if p.exists():
            obs[f] = json.load(open(p))["variants"]["cohort"]["auc_roc"]["mean"]
    summary = {
        "task": "T2.10b",
        "question": ("upper bound on how much of the within-cohort scramble "
                     "residual (split, t0) cell composition can explain"),
        "model": "no features; score = base rate of the row's own (split, t0) cell",
        "fields": {f: {**per[f], "observed_e9a_cohort_auc_roc": obs.get(f)}
                   for f in per},
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print("\n  field       observed e9a cohort   composition-only bound")
    for f in FIELDS:
        if f in per:
            print(f"  {f:10} {obs.get(f, float('nan')):>18.4f}   "
                  f"{per[f]['composition_only_auc_roc_vs_scrambled']['mean']:>10.4f}")
    print(f"\n[T2.10b] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
