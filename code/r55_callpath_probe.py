#!/usr/bin/env python3
"""Measure the call-path dependence of M3s_gbdt, and write it to a file.

WHY THIS EXISTS. The paper already states that the gradient-boosted student-only
rung returns a different number on identical inputs and identical seeds
according to what ran earlier in the same Python process, and gives the three
values and the six prefixes that isolate it. That statement had no result file
behind it, which is the one thing every other number in the paper has. The
construction-effects table quotes the magnitude, so the magnitude has to be
readable from an artifact rather than from a sentence.

WHAT IS MEASURED. One cell: economics, theta 0.2, the four-feature student-only
floor's M3s_gbdt rung, ten seeds, on the legacy TF-IDF footing that every
recorded floor uses. Everything is held fixed except what runs before the fit.

HOW. The effect is a property of process state, so each prefix runs in its own
subprocess: this file re-invokes itself with --prefix NAME, the child executes
that prefix and then the identical fit, and the parent collects the values. A
single process could not measure this, because the first prefix would
contaminate every later one, which is the effect itself.

The prefixes are the paper's six, plus the empty one:
  none        no prefix at all
  dense_logit dense logistic regression, which enters multithreaded BLAS
  lbfgsb      a scipy L-BFGS-B call, same path by a different door
  matmul      a bare A @ A, the path with nothing else attached
  scaler      StandardScaler alone
  liblinear   a sparse liblinear fit
  gbdt_first  a HistGradientBoosting fit on unrelated data

Reads only frozen artifacts. Writes results/revision/T2_13_callpath/callpath.json.

  python code/r55_callpath_probe.py
  python code/r55_callpath_probe.py --prefix scaler     (one child, prints one line)
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_13_callpath"
FIELD = "econ"
THETA = 0.20
FLOOR = 0.0013          # DETERMINISM_MEASURED.json, max drift over five repeated cells
PREFIXES = ["none", "dense_logit", "lbfgsb", "matmul", "scaler", "liblinear",
            "gbdt_first"]


def run_prefix(name):
    """Execute one prefix, then the identical M3s_gbdt fit. Returns the mean."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))
    import config as C
    from utils import data as D
    from utils import stats as S
    from e14_self_persistence import fit_gbdt, student_tfidf
    from r1_theta_sweep import student_features_no_store

    rng = np.random.default_rng(0)
    A = rng.random((256, 256))
    if name == "dense_logit":
        from sklearn.linear_model import LogisticRegression
        y = (rng.random(256) > 0.5).astype(int)
        LogisticRegression(max_iter=50).fit(A, y)
    elif name == "lbfgsb":
        from scipy.optimize import minimize
        minimize(lambda v: float(v @ v), rng.random(64), method="L-BFGS-B")
    elif name == "matmul":
        _ = A @ A
    elif name == "scaler":
        from sklearn.preprocessing import StandardScaler
        StandardScaler().fit(A)
    elif name == "liblinear":
        from scipy import sparse as sp
        from sklearn.linear_model import LogisticRegression
        y = (rng.random(256) > 0.5).astype(int)
        LogisticRegression(solver="liblinear", max_iter=50).fit(sp.csr_matrix(A), y)
    elif name == "gbdt_first":
        from sklearn.ensemble import HistGradientBoostingClassifier
        y = (rng.random(256) > 0.5).astype(int)
        HistGradientBoostingClassifier(max_iter=5).fit(A, y)

    df = D.load_dataset()
    probe = D.temporal_split(df.copy())
    X_st = student_tfidf(probe, legacy=True)      # the recorded footing
    d = df.copy()
    d["y"] = (d.late_overlap > THETA).astype(int)
    ds = D.temporal_split(d)
    F = student_features_no_store(ds)
    masks = {k: (ds.split == k).values for k in ("train", "val", "test")}
    y = {k: ds.loc[m, "y"].values for k, m in masks.items()}
    X = np.hstack([F.values, X_st.toarray()]).astype(np.float32)
    runs = [fit_gbdt(X[masks["train"]], y["train"], X[masks["val"]], y["val"],
                     X[masks["test"]], y["test"], seed=s)[0] for s in C.SEEDS]
    return float(np.mean([r["auc_pr"] for r in runs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", choices=PREFIXES, default=None)
    a = ap.parse_args()
    if a.prefix:
        print(f"VALUE {a.prefix} {run_prefix(a.prefix):.6f}")
        return 0

    env = dict(os.environ, DATASET=FIELD,
               DATASET_PATH=f"data/clean_dataset_{FIELD}.parquet")
    vals = {}
    for name in PREFIXES:
        r = subprocess.run([sys.executable, str(Path(__file__)), "--prefix", name],
                           cwd=ROOT, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        line = next((l for l in r.stdout.splitlines() if l.startswith("VALUE ")),
                    None)
        if not line:
            raise SystemExit(f"r55: prefix {name} produced no value:\n"
                             f"{r.stdout[-500:]}\n{r.stderr[-500:]}")
        vals[name] = float(line.split()[2])
        print(f"[r55] {name:12s} {vals[name]:.6f}", flush=True)

    lo, hi = min(vals.values()), max(vals.values())
    branches = {}
    for k, v in vals.items():
        branches.setdefault(round(v, 4), []).append(k)
    out = {
        "task": "T2.13",
        "what": ("call-path dependence of the M3s_gbdt student-only rung: the "
                 "same cell, the same inputs and the same ten seeds, differing "
                 "only in what ran earlier in the same Python process"),
        "cell": {"field": FIELD, "theta": THETA, "rung": "M3s_gbdt",
                 "features": 4, "tfidf_footing": "legacy (all rows)",
                 "seeds": 10},
        "values_by_prefix": {k: round(v, 6) for k, v in vals.items()},
        "distinct_branches": {str(k): v for k, v in sorted(branches.items())},
        "n_branches": len(branches),
        "spread": round(hi - lo, 6),
        "determinism_floor": FLOOR,
        "floor_source": ("results/revision/T2_1_strict_contract/chemistry/"
                         "DETERMINISM_MEASURED.json, max drift over five cells "
                         "run twice"),
        "ratio_to_floor": round((hi - lo) / FLOOR, 1),
        "why_not_the_floor": ("the floor is variation between two runs of the "
                              "same call; this is variation between call paths, "
                              "so folding one into the other would understate "
                              "both"),
        "published_branch": ("r1_theta_sweep.py fits the deterministic M1 "
                             "reference before any student-only design, so the "
                             "recorded floors are the branch that prefix "
                             "produces"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "callpath.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[r55] {len(branches)} distinct branches, spread {hi - lo:.4f}, "
          f"{out['ratio_to_floor']}x the floor")
    print(f"[r55] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
