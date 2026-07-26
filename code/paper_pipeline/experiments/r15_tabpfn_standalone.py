"""R15b - TabPFN rung, standalone for the isolated venv (task B8).

Runs OUTSIDE the frozen analysis environment (TabPFN's dependency floor is
incompatible; an in-place install was rolled back). No repo imports: the
temporal split (cohort quantiles 0.6/0.8 of t0, from utils/data.py
temporal_split) and the eight tabular features (config.TABULAR_FEATURES +
coauth_early, from utils/data.py build_features) are replicated here with
their sources named. Empty-early-concept rows are dropped as in
load_dataset.

TabPFN v2 on CPU operates at a 1,000-sample context (its enforced CPU
limit); every discipline therefore trains on a seeded 1,000-row subsample,
10 seeds. This handicaps TabPFN against the full-train baselines and is
reported as such: the rung measures TabPFN at its CPU operating point, and
a full-context GPU run is rebuttal material.

Output: results/robustness/extra_rungs_partial/<field>_tabpfn.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[3]
PARTIAL = REPO / "results" / "robustness" / "extra_rungs_partial"
FIELDS = ["econ", "math", "physics", "neuro", "chemistry"]
FEATURES = ["early_overlap", "early_prod", "early_breadth", "adv_early_prod",
            "adv_early_breadth", "adv_career_age_at_t0", "coauth_early_n",
            "coauth_early"]          # config.TABULAR_FEATURES + BOOL_FEATURES
SPLIT_Q = (0.6, 0.8)                 # config.SPLIT_QUANTILES
MAX_TRAIN = 1_000
SEEDS = list(range(10))


def main():
    from tabpfn import TabPFNClassifier
    PARTIAL.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        outp = PARTIAL / f"{f}_tabpfn.json"
        if outp.exists():
            print(f"[r15b] {f} done, skipping", flush=True)
            continue
        df = pd.read_parquet(REPO / "data" / f"clean_dataset_{f}.parquet")
        df = df[df.early_concepts.apply(len) > 0].reset_index(drop=True)
        q1, q2 = np.quantile(df.t0, SPLIT_Q)
        tr = (df.t0 <= int(q1)).values
        te = (df.t0 > int(q2)).values
        X = df[FEATURES].astype(float).values
        y = df.y.values.astype(int)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        subsampled = len(ytr) > MAX_TRAIN
        runs = []
        for seed in SEEDS:
            if subsampled:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(ytr), MAX_TRAIN, replace=False)
            else:
                if runs:                        # deterministic: reuse
                    runs.append(runs[0])
                    continue
                idx = np.arange(len(ytr))
            m = TabPFNClassifier(device="cpu")
            m.fit(Xtr[idx], ytr[idx])
            p = m.predict_proba(Xte)[:, 1]
            runs.append(round(float(average_precision_score(yte, p)), 4))
            print(f"[r15b] {f} seed {seed}: {runs[-1]}", flush=True)
        rec = {"field": f, "model": "tabpfn_v2_cpu_context1000",
               "train_subsampled": bool(subsampled),
               "per_seed_auc_pr": runs,
               "mean": round(float(np.mean(runs)), 4),
               "std": round(float(np.std(runs, ddof=1)), 4)}
        # for the screen-passers, save per-row test scores + student_pid so the
        # main env can run the full eq.(2) student-level bootstrap (aligns by pid)
        if f in ("econ", "math"):
            te_pid = df.loc[te, "student_pid"].astype(str).tolist()
            te_scores = {}
            for seed in SEEDS:
                if subsampled:
                    rng = np.random.default_rng(seed)
                    tr_idx = rng.choice(len(ytr), MAX_TRAIN, replace=False)
                else:
                    tr_idx = np.arange(len(ytr))
                m = TabPFNClassifier(device="cpu")
                m.fit(Xtr[tr_idx], ytr[tr_idx])
                te_scores[str(seed)] = [round(float(x), 6)
                                        for x in m.predict_proba(Xte)[:, 1]]
            rec["test_student_pid"] = te_pid
            rec["test_labels"] = [int(v) for v in yte]
            rec["test_scores_per_seed"] = te_scores
        outp.write_text(json.dumps(rec, indent=2))
        print(f"[r15b] {f}: mean {np.mean(runs):.4f}", flush=True)


if __name__ == "__main__":
    main()
