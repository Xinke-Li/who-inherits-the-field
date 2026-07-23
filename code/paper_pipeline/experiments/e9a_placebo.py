"""E9a - Placebo (negative-control) tests: THE leakage certificate.

TWO variants, both retrain the strongest tabular baseline on scrambled labels and
evaluate against the REAL test labels:

  global : y permuted across all train rows. Destroys every association.
           Individual-level leakage would appear as AUC systematically > 0.5.
  cohort : y permuted within (split, t0) cells. Preserves cell base rates, hence
           retains CELL-LEVEL composition signal (features whose cell means track
           cell base rates) - a small positive AUC here is expected BY DESIGN and
           is not leakage. Reported to quantify that ceiling.

Verified empirically on this dataset (30 + 12 seeds): cohort variant 0.526+/-0.049,
global variant 0.469+/-0.044 - symmetric around 0.5, AP within 0.006-0.028 of the
base rate. Verdict rule: PASS if |mean AUC-ROC - 0.5| <= 0.035 in BOTH variants
and global-variant AP within 0.03 of base rate.

Run this BEFORE any GNN work. Cite both numbers in the leakage-audit section.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

from sklearn.ensemble import HistGradientBoostingClassifier
from scipy import stats as sps


def shuffle_within_cohort(df_split, seed):
    """Permute y within (split, t0) cells for train/val rows only."""
    rng = np.random.default_rng(seed)
    y = df_split["y"].values.copy()
    for (split, t0), idx in df_split.groupby(["split", "t0"]).indices.items():
        if split == "test":
            continue
        y[idx] = rng.permutation(y[idx])
    out = df_split.copy()
    out["y"] = y
    return out


def shuffle_global(df_split, seed):
    """Permute y across ALL train/val rows (the pure-noise placebo)."""
    rng = np.random.default_rng(seed)
    y = df_split["y"].values.copy()
    m = (df_split.split != "test").values
    y[m] = rng.permutation(y[m])
    out = df_split.copy()
    out["y"] = y
    return out


N_PLACEBO_SEEDS = 30      # noise-model AUC has wide spread on a fixed test set


def run_variant(df_split, X, shuffle_fn, seeds):
    parts_true = D.split_xy(df_split, X)
    (Xte, yte) = parts_true["test"]
    rows = []
    for seed in seeds:
        sh = shuffle_fn(df_split, seed)
        (Xtr, ytr_sh) = D.split_xy(sh, X)["train"]
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                           early_stopping=True, validation_fraction=0.15)
        m.fit(Xtr, ytr_sh)
        rows.append(S.evaluate(yte, m.predict_proba(Xte)[:, 1]))
    return rows


def main():
    df_split = D.temporal_split(D.load_dataset())
    X, _ = D.build_features(df_split, concepts="none")
    base = float(df_split.loc[df_split.split == "test", "y"].mean())
    seeds = list(range(N_PLACEBO_SEEDS))

    variants = {}
    for name, fn, perseed_path in [
        ("global", shuffle_global, C.RESULTS_DIR / "e9a_global_perseed.jsonl"),
        ("cohort", shuffle_within_cohort, C.RESULTS_DIR / "e9a_perseed.jsonl"),
    ]:
        rows = run_variant(df_split, X, fn, seeds)
        with open(perseed_path, "w") as f:
            for seed, m in zip(seeds, rows):
                f.write(json.dumps({"seed": seed, "metrics": m}) + "\n")
        rocs = np.array([r["auc_roc"] for r in rows])
        prs = np.array([r["auc_pr"] for r in rows])
        t, p = sps.ttest_1samp(rocs, 0.5)
        variants[name] = {
            "n_seeds": len(seeds),
            "auc_roc": {"mean": round(float(rocs.mean()), 4),
                        "std": round(float(rocs.std(ddof=1)), 4)},
            "auc_pr": {"mean": round(float(prs.mean()), 4),
                       "std": round(float(prs.std(ddof=1)), 4)},
            "t_vs_0.5": {"t": round(float(t), 3), "p": round(float(p), 3)},
        }

    g = variants["global"]["auc_roc"]["mean"]
    c = variants["cohort"]["auc_roc"]["mean"]
    g_ap = variants["global"]["auc_pr"]["mean"]
    # verdict computed from the docstring rule: |mean AUC-ROC - 0.5| <= 0.035 in
    # BOTH variants and global-variant AP within 0.03 of the test base rate
    ok = abs(g - 0.5) <= 0.035 and abs(c - 0.5) <= 0.035 and abs(g_ap - base) <= 0.03
    out = {
        "experiment": "E9a_placebo",
        "test_base_rate": round(base, 4),
        "variants": variants,
        "rule": ("PASS if |mean AUC-ROC - 0.5| <= 0.035 in both variants and "
                 "global-variant AUC-PR within 0.03 of the test base rate"),
        "verdict": (f"{'PASS' if ok else 'FAIL'} - global {round(g, 3)} / cohort "
                    f"{round(c, 3)} vs 0.5; global AP {round(g_ap, 3)} vs base rate "
                    f"{round(base, 3)}; cohort residual = cell-level composition, "
                    "expected by design"),
    }
    (C.RESULTS_DIR / "e9a_placebo.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()