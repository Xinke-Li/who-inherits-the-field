"""Statistical machinery shared by all experiments.

Design principles (pre-registered in the analysis plan; rules restated in each experiment script's docstring):
- every stochastic result: 10 seeds, mean +/- std;
- comparisons vs a baseline model: PAIRED per-seed tests (Wilcoxon signed-rank),
  Benjamini-Hochberg correction across the comparison family;
- headline metrics carry bootstrap 95% CIs on the test set.
"""
import numpy as np
from scipy import stats as sps
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_recall_curve)


# ---------- metrics ----------
def evaluate(y_true, scores, threshold=None):
    """AUC-PR (primary), AUC-ROC, F1 at the given (val-optimized) threshold."""
    out = {
        "auc_pr": float(average_precision_score(y_true, scores)),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "base_rate": float(np.mean(y_true)),
    }
    if threshold is not None:
        out["f1"] = float(f1_score(y_true, scores >= threshold))
        out["threshold"] = float(threshold)
    return out


def best_f1_threshold(y_val, val_scores):
    """Pick the F1-maximizing threshold on the VALIDATION set only."""
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[max(np.nanargmax(f1[:-1]), 0)])


def bootstrap_ci(y_true, scores, metric="auc_pr", n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI over test rows."""
    fn = average_precision_score if metric == "auc_pr" else roc_auc_score
    rng = np.random.default_rng(seed)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    n, vals = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue  # degenerate resample
        vals.append(fn(y_true[idx], scores[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------- multi-seed comparison ----------
def summarize_seeds(per_seed: list[dict], keys=("auc_pr", "auc_roc", "f1")):
    """[{'auc_pr':..,'auc_roc':..}, ...] per seed -> {'auc_pr': (mean, std), ...}"""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in per_seed if k in r])
        if len(v):
            out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1) if len(v) > 1 else 0.0),
                      "n_seeds": int(len(v))}
    return out


def paired_wilcoxon(a: list[float], b: list[float]):
    """Paired (per-seed) Wilcoxon signed-rank test, two-sided.
    Returns dict with p-value, mean difference, and per-seed differences."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert len(a) == len(b), "paired test needs equal seed counts"
    d = a - b
    if np.allclose(d, 0):
        return {"p": 1.0, "mean_diff": 0.0, "diffs": d.tolist()}
    stat, p = sps.wilcoxon(a, b)
    return {"p": float(p), "mean_diff": float(d.mean()),
            "std_diff": float(d.std(ddof=1)), "diffs": d.tolist()}


def bh_correction(pvals: list[float], alpha=0.05):
    """Benjamini-Hochberg. Returns (adjusted p-values, reject flags)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj_full = np.empty(m)
    adj_full[order] = np.clip(adj, 0, 1)
    return adj_full.tolist(), (adj_full <= alpha).tolist()


def cohens_d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("inf") * np.sign(d.mean())
