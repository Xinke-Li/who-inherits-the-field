"""E17 (cache edition) - swap-label control on the r5 author cache (task B2).

POST-HOC ROBUSTNESS CHECK, not part of the pre-registered protocol. E10 swaps
only the FEATURE side (early overlap against a placebo advisor) while the
label stays a function of the true advisor profile, so part of E10's gap is
mechanical alignment between a feature and a label that share A_true. This
control swaps BOTH sides onto the placebo advisor: the placebo system's
feature is J(C_early, A_p) and its label is 1[J(C_late, A_p) > theta'], with
theta' calibrated on TRAIN so the placebo base rate matches the true system's,
making the two AUC-PRs comparable at one operating point. The advisor-specific
excess is the true-system AUC-PR minus the placebo-system AUC-PR.

Design restated from the original e17_swaplabel.py (source of the decision
rule and the calibration rule); implementation differences, both required by
the data situation:
  * every profile (student early, student late, advisor) is REBUILT from the
    r5 OpenAlex cache at the builder view (score >= 0.3, first three, top-10),
    so both systems live in one provenance family (r6 convention: rebuilt
    quantities are never compared against frozen ones); the frozen M1 numbers
    are reported alongside as reference only;
  * the excess carries a paired student-level bootstrap CI (2000 draws over
    test students, pooled over the 10 placebo redraws by averaging the
    per-redraw difference within each draw - e12's F4 pooling), in addition
    to the redraw-percentile interval of the original design.

PRE-REGISTERED DECISION RULE (from e17_swaplabel.py, unchanged): the true
pairing carries advisor-specific predictive structure beyond mechanical
label-feature sharing iff the true-system AUC-PR exceeds the swap-label
placebo system with an interval that excludes zero.

Usage:
  DATASET=<field> DATASET_PATH=data/clean_dataset_<field>.parquet \
      python code/paper_pipeline/experiments/e17_swap_label_cache.py
  python code/paper_pipeline/experiments/e17_swap_label_cache.py --merge

Output: results/robustness/e17_partial/<field>.json,
        results/robustness/e17_swap_label.json (--merge)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D

from r6_topk_sweep import (BUILDER_MIN_SCORE, builder_view, load_cache_for,
                           profile_counts, topk_set)
from e14_self_persistence import jaccard

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
from e12_corrected_aggregation import fast_auc_pr

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "results" / "robustness"
PARTIAL_DIR = OUT_DIR / "e17_partial"
FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]
COHORT_WINDOW = 3          # e10/e17 convention
K = 10
N_BOOT = 2000


def calibrate_theta(lo_train, target_rate):
    """e17_swaplabel.calibrate_theta verbatim: fixed grid, train rows only."""
    grid = np.linspace(0.0, 0.9, 91)
    rates = np.array([(lo_train > th).mean() for th in grid])
    j = int(np.argmin(np.abs(rates - target_rate)))
    return float(grid[j]), float(rates[j])


def calibrate_exact(lo_all, lo_train, target_rate, rng):
    """Exact base-rate matching for a DISCRETE overlap distribution: the
    top-10 Jaccard takes atomic values, so a hard threshold can miss the
    target rate by a wide margin (observed: econ placebo 0.07 vs target
    0.20). Fix: threshold at the atom that brackets the target and label the
    boundary atom positive with the randomization probability q that makes
    the TRAIN rate hit the target exactly. Returns (y_p over all rows,
    theta_lo, q). Train-only calibration; the same theta and q apply to
    every row."""
    atoms = np.unique(lo_train)
    # rate(th) = P(lo > th); find the atom a such that
    # P(lo > a) <= target <= P(lo >= a); randomize on lo == a
    p_gt = np.array([(lo_train > a).mean() for a in atoms])
    p_ge = np.array([(lo_train >= a).mean() for a in atoms])
    ok = np.where((p_gt <= target_rate) & (p_ge >= target_rate))[0]
    if len(ok) == 0:
        a = atoms[int(np.argmin(np.abs(p_gt - target_rate)))]
        q = 0.0
    else:
        a = atoms[ok[0]]
        mass = (lo_train == a).mean()
        q = 0.0 if mass == 0 else (target_rate - (lo_train > a).mean()) / mass
    y = (lo_all > a).astype(int)
    at = lo_all == a
    y[at] = (rng.random(at.sum()) < q).astype(int)
    return y, float(a), float(q)


def main_field():
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    df = D.load_dataset()
    ds = D.temporal_split(df)

    aids = set(df.st_openalex_id) | set(df.adv_openalex_id)
    print(f"[e17] {field}: loading cache for {len(aids)} authors", flush=True)
    cache = load_cache_for(aids)
    view = {a: builder_view(cache[a], BUILDER_MIN_SCORE) for a in aids if a in cache}

    early_sets, late_sets, adv_sets, keep = [], [], [], []
    for i, r in enumerate(df.itertuples()):
        sw, aw = view.get(r.st_openalex_id), view.get(r.adv_openalex_id)
        if not sw or not aw:
            keep.append(False)
            early_sets.append(None); late_sets.append(None); adv_sets.append(None)
            continue
        keep.append(True)
        e_cnt, _ = profile_counts(sw, r.t0, r.t0 + C.EARLY_YEARS)
        l_cnt, _ = profile_counts(sw, r.t0 + C.EARLY_YEARS + 1, r.t0 + C.LATE_YEARS)
        a_cnt, _ = profile_counts(aw, None, r.t0 + C.EARLY_YEARS)
        early_sets.append(topk_set(e_cnt, K))
        late_sets.append(topk_set(l_cnt, K))
        adv_sets.append(topk_set(a_cnt, K))
    keep = np.array(keep)
    print(f"[e17] {field}: {keep.sum()}/{len(df)} rows with full cache coverage",
          flush=True)

    idx_rows = np.where(keep)[0]
    t0 = df.t0.values[idx_rows]
    adv_pid = df.advisor_pid.values[idx_rows]
    E = [early_sets[i] for i in idx_rows]
    L = [late_sets[i] for i in idx_rows]
    A = [adv_sets[i] for i in idx_rows]
    split = ds.split.values[idx_rows]
    is_test, is_train = split == "test", split == "train"
    n = len(idx_rows)

    # true system, rebuilt family
    eo_true = np.array([jaccard(E[i], A[i]) for i in range(n)])
    lo_true = np.array([jaccard(L[i], A[i]) for i in range(n)])
    y_true = (lo_true > C.JACCARD_THETA).astype(int)
    base_rate = float(y_true.mean())
    frozen_dev = float(np.abs(lo_true.round(4)
                              - df.late_overlap.values[idx_rows]).mean())
    true_auc = fast_auc_pr(y_true[is_test], eo_true[is_test])
    from sklearn.metrics import roc_auc_score as _roc
    true_roc = float(_roc(y_true[is_test], eo_true[is_test]))

    # placebo systems: 10 cohort-matched redraws
    per_seed, eo_ps, y_ps = [], [], []
    order = np.argsort(t0, kind="stable")
    t0_sorted = t0[order]
    for seed in C.SEEDS:
        rng = np.random.default_rng(seed)
        eo_p = np.empty(n); lo_p = np.empty(n)
        for i in range(n):
            lo_i = np.searchsorted(t0_sorted, t0[i] - COHORT_WINDOW, "left")
            hi_i = np.searchsorted(t0_sorted, t0[i] + COHORT_WINDOW, "right")
            pool = order[lo_i:hi_i]
            pool = pool[adv_pid[pool] != adv_pid[i]]
            if len(pool) == 0:
                pool = np.where(adv_pid != adv_pid[i])[0]
            j = int(pool[rng.integers(len(pool))])
            eo_p[i] = jaccard(E[i], A[j])
            lo_p[i] = jaccard(L[i], A[j])
        theta_p, tr_rate = calibrate_theta(lo_p[is_train], base_rate)
        y_grid = (lo_p > theta_p).astype(int)
        auc_grid = fast_auc_pr(y_grid[is_test], eo_p[is_test])
        # exact randomized calibration (primary for the paper: the grid rule
        # cannot hit the target on a discrete overlap distribution)
        y_p, theta_lo, q = calibrate_exact(lo_p, lo_p[is_train], base_rate, rng)
        placebo_auc = fast_auc_pr(y_p[is_test], eo_p[is_test])
        from sklearn.metrics import roc_auc_score
        placebo_roc = float(roc_auc_score(y_p[is_test], eo_p[is_test]))
        eo_ps.append(eo_p[is_test]); y_ps.append(y_p[is_test])
        per_seed.append({"seed": seed,
                         "grid_theta_p": round(theta_p, 3),
                         "grid_placebo_auc_pr": round(auc_grid, 4),
                         "grid_placebo_base_rate_test": round(float(y_grid[is_test].mean()), 4),
                         "exact_theta_atom": round(theta_lo, 4), "exact_q": round(q, 4),
                         "placebo_base_rate_train": round(float(y_p[is_train].mean()), 4),
                         "placebo_base_rate_test": round(float(y_p[is_test].mean()), 4),
                         "placebo_auc_pr": round(placebo_auc, 4),
                         "placebo_auc_roc": round(placebo_roc, 4),
                         "excess": round(true_auc - placebo_auc, 4)})
        print(f"[e17] {field} seed {seed}: exact base "
              f"{y_p[is_test].mean():.3f} placebo={placebo_auc:.4f} "
              f"excess={true_auc - placebo_auc:+.4f}", flush=True)

    # paired student-level bootstrap over test rows, pooled over redraws
    yt, st = y_true[is_test], eo_true[is_test]
    m = len(yt)
    rng = np.random.default_rng(0)
    pooled = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, m, m)
        if yt[idx].min() == yt[idx].max():
            pooled[b] = np.nan
            continue
        t_ap = fast_auc_pr(yt[idx], st[idx])
        diffs = []
        for r_ in range(len(C.SEEDS)):
            ypb = y_ps[r_][idx]
            if ypb.min() == ypb.max():
                continue
            diffs.append(t_ap - fast_auc_pr(ypb, eo_ps[r_][idx]))
        pooled[b] = float(np.mean(diffs)) if diffs else np.nan
    pooled = pooled[~np.isnan(pooled)]
    ci_student = [round(float(np.percentile(pooled, 2.5)), 4),
                  round(float(np.percentile(pooled, 97.5)), 4)]

    ex = np.array([r["excess"] for r in per_seed])
    ci_redraw = [round(float(np.percentile(ex, 2.5)), 4),
                 round(float(np.percentile(ex, 97.5)), 4)]
    frozen_m1 = None
    e1p = REPO / "results" / f"results_{field}" / "e1_baselines.json"
    if e1p.exists():
        frozen_m1 = json.loads(e1p.read_text())["summary"]["M1_logit_overlap"]["auc_pr"]["mean"]

    out = {
        "experiment": "E17_swap_label_control", "field": field,
        "provenance": "rebuilt family (r5 cache, builder view); frozen M1 for reference only",
        "n_rows": int(n), "n_test": int(m),
        "base_rate_rebuilt": round(base_rate, 4),
        "mean_abs_dev_lo_vs_frozen": round(frozen_dev, 5),
        "true_system_auc_pr": round(true_auc, 4),
        "true_system_auc_roc": round(true_roc, 4),
        "placebo_auc_roc_mean": round(float(np.mean(
            [r["placebo_auc_roc"] for r in per_seed])), 4),
        "excess_roc_mean": round(true_roc - float(np.mean(
            [r["placebo_auc_roc"] for r in per_seed])), 4),
        "frozen_M1_reference": frozen_m1,
        "placebo_auc_pr_mean": round(float(np.mean([r["placebo_auc_pr"] for r in per_seed])), 4),
        "excess_mean": round(float(ex.mean()), 4),
        "excess_ci95_redraw": ci_redraw,
        "excess_ci95_student_pooled": ci_student,
        "excess_excludes_zero": bool(ci_student[0] > 0),
        "per_seed": per_seed,
    }
    (PARTIAL_DIR / f"{field}.json").write_text(json.dumps(out, indent=2))
    print(f"[e17] {field}: true {out['true_system_auc_pr']} placebo "
          f"{out['placebo_auc_pr_mean']} excess {out['excess_mean']} "
          f"studentCI {ci_student}", flush=True)


def main_merge():
    merged = {"experiment": "E17_swap_label_control",
              "rule": ("advisor-specific structure beyond mechanical label-feature "
                       "sharing iff excess > 0 with the student-level CI excluding zero"),
              "fields": {}}
    for f in FIELDS:
        merged["fields"][f] = json.loads((PARTIAL_DIR / f"{f}.json").read_text())
    merged["all_positive"] = all(v["excess_excludes_zero"]
                                 for v in merged["fields"].values())
    (OUT_DIR / "e17_swap_label.json").write_text(json.dumps(merged, indent=2))
    for f in FIELDS:
        v = merged["fields"][f]
        print(f"{f:10} excess {v['excess_mean']:+.4f} studentCI "
              f"{v['excess_ci95_student_pooled']} excludes0="
              f"{v['excess_excludes_zero']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        main_merge()
    else:
        main_field()
