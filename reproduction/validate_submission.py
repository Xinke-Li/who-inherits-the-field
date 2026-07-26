#!/usr/bin/env python3
"""T4.1: validate a benchmark submission before it is compared to anything.

The benchmark's whole claim is that a graph gain has to clear a calibrated
ceiling under a protocol that is fixed in advance. That is worth nothing if a
submitter can quietly change the split, report the best of thirty seeds, or
select a checkpoint on the test cohort. This script refuses a submission that
does any of those, and it refuses before computing a score, so a rejected
submission never produces a number anyone can quote.

Six checks, all blocking:

  1  schema        the required fields are present and typed
  2  seeds         exactly ten seeds, 0 through 9, no duplicates
  3  frozen split  the submitted test student_pid list equals the frozen
                   temporal test split of that discipline, in order
  4  labels        the submitted test labels equal the frozen labels
  5  one test eval selection is declared and its split is not "test"; a
                   per-seed test score array is present for exactly the ten
                   declared seeds and nothing else
  6  scores        finite, inside [0, 1], one per test student, per seed

A submission that passes is then scored against the validation-symmetric
ceiling M5$'$ under eq. (2): the ten-seed mean gap, the paired student-level
bootstrap over 2000 draws, and the three gates. Passing validation is not
passing the gates; the two are reported separately on purpose.

  python reproduction/validate_submission.py --submission my_run.json
  python reproduction/validate_submission.py --demo-from results/results_econ/results_hgt --field econ

The demo mode builds a submission in memory from a directory of the paper's own
per-seed artifacts and validates it, which is how this script is tested: if the
validator disagrees with the paper's own graph runs, the validator is wrong.

Template: reproduction/submission_template.json
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
SEEDS = list(range(10))
N_BOOT = 2000
REQUIRED = ["benchmark", "field", "model_name", "seeds", "selection",
            "test_student_pid", "test_labels", "test_scores_per_seed"]


class Reject(Exception):
    """A validation failure. Raised, never returned, so no caller can ignore
    it and go on to compute a score."""


def frozen_split(field):
    os.environ.setdefault("DATASET", field)
    os.environ["DATASET"] = field
    os.environ["DATASET_PATH"] = str(ROOT / "data" / f"clean_dataset_{field}.parquet")
    from utils import data as D
    ds = D.temporal_split(D.load_dataset())
    te = ds[ds.split == "test"]
    return ds, te.student_pid.astype(str).tolist(), te.y.values.astype(int)


def validate(sub, field=None):
    """Six blocking checks. Returns the frame and the frozen arrays."""
    checks = []

    missing = [k for k in REQUIRED if k not in sub]
    if missing:
        raise Reject(f"check 1 schema: missing fields {missing}")
    if sub.get("benchmark") != "research-area-persistence-v1":
        raise Reject(f"check 1 schema: benchmark is "
                     f"{sub.get('benchmark')!r}, expected "
                     f"'research-area-persistence-v1'")
    f = field or sub["field"]
    if f not in FIELDS:
        raise Reject(f"check 1 schema: field {f!r} is not one of {FIELDS}")
    checks.append("1 schema")

    seeds = sub["seeds"]
    if sorted(seeds) != SEEDS:
        raise Reject(f"check 2 seeds: got {sorted(seeds)}, the protocol fixes "
                     f"exactly {SEEDS}. Reporting a subset is selection on the "
                     f"test cohort by another name.")
    checks.append("2 seeds")

    ds, pid, y = frozen_split(f)
    if list(map(str, sub["test_student_pid"])) != pid:
        n_sub, n_frozen = len(sub["test_student_pid"]), len(pid)
        raise Reject(f"check 3 frozen split: the submitted test cohort does not "
                     f"equal the frozen temporal test split of {f} "
                     f"({n_sub} rows against {n_frozen}). The split is part of "
                     f"the benchmark, not a modelling choice.")
    checks.append("3 frozen split")

    if [int(v) for v in sub["test_labels"]] != list(y):
        raise Reject("check 4 labels: the submitted test labels do not equal "
                     "the frozen labels for this split.")
    checks.append("4 labels")

    sel = sub["selection"]
    if not isinstance(sel, dict) or "split" not in sel:
        raise Reject("check 5 one test evaluation: 'selection' must be an "
                     "object declaring at least {'split': ..., 'metric': ...}.")
    if str(sel["split"]).lower() == "test":
        raise Reject("check 5 one test evaluation: selection.split is 'test'. "
                     "The test cohort is evaluated once, after training and "
                     "after every choice has been made on train or validation.")
    got = sorted(int(k) for k in sub["test_scores_per_seed"])
    if got != SEEDS:
        raise Reject(f"check 5 one test evaluation: score arrays present for "
                     f"seeds {got}, declared {SEEDS}. Extra arrays mean the "
                     f"test cohort was scored more than the declared number of "
                     f"times.")
    checks.append("5 one test evaluation")

    n = len(pid)
    for s in SEEDS:
        a = np.asarray(sub["test_scores_per_seed"][str(s)]
                       if str(s) in sub["test_scores_per_seed"]
                       else sub["test_scores_per_seed"][s], float)
        if a.shape != (n,):
            raise Reject(f"check 6 scores: seed {s} has {a.shape[0] if a.ndim else '?'} "
                         f"scores, expected {n}.")
        if not np.isfinite(a).all():
            raise Reject(f"check 6 scores: seed {s} contains a non-finite value.")
        if a.min() < 0.0 or a.max() > 1.0:
            raise Reject(f"check 6 scores: seed {s} has values outside [0, 1] "
                         f"({a.min():.4g} to {a.max():.4g}).")
    checks.append("6 scores")
    return ds, pid, y, checks


def score(sub, ds, y):
    """eq. (2) against M5', after validation and only after."""
    import e12_corrected_aggregation as E12
    from scipy.stats import wilcoxon
    from utils import data as D
    from utils import stats as S

    field = sub["field"]
    Xt, _ = D.build_features(ds, concepts="none")
    nfa = D.build_nfa_features(ds)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    p5 = D.split_xy(ds, X5)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = p5["train"], p5["val"], p5["test"]

    stored = json.loads((ROOT / "results" / f"results_{field}" /
                         "e12_corrected_vs_m5.json").read_text())
    ref = stored["ceilings"]["per_seed"]["M5_prime_val_symmetric"]
    m5p, m5p_ap, devs = {}, [], []
    for s in SEEDS:
        p, _, _, _ = E12.fit_val_symmetric(Xtr, ytr, Xva, yva, Xte, s)
        m5p[s] = np.asarray(p, float)
        ap = E12.fast_auc_pr(yte, p)
        m5p_ap.append(ap)
        devs.append(abs(ap - ref[s]))
    if max(devs) >= 5e-4:
        raise Reject(f"the ceiling does not reproduce in this environment "
                     f"(max per-seed deviation {max(devs):.6f}); no verdict "
                     f"from this run would be trustworthy.")

    g = {s: np.asarray(sub["test_scores_per_seed"][str(s)], float) for s in SEEDS}
    g_ap = [E12.fast_auc_pr(yte, g[s]) for s in SEEDS]

    rng = np.random.default_rng(0)
    n = len(yte)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    pooled = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        i = idx[b]
        yb = yte[i]
        if yb.sum() in (0, len(yb)):
            continue
        pooled[b] = float(np.mean([E12.fast_auc_pr(yb, g[s][i])
                                   - E12.fast_auc_pr(yb, m5p[s][i])
                                   for s in SEEDS]))
    ci = [round(float(np.nanpercentile(pooled, 2.5)), 4),
          round(float(np.nanpercentile(pooled, 97.5)), 4)]
    d = float(np.mean(g_ap) - np.mean(m5p_ap))
    p_raw = float(wilcoxon(np.array(g_ap) - np.array(m5p_ap)).pvalue)
    p_bh = float(S.bh_correction([p_raw])[0][0])
    gates = {"seed_mean_gt_ceiling": bool(np.mean(g_ap) > np.mean(m5p_ap)),
             "p_adj_lt_0.05": bool(p_bh < 0.05),
             "student_ci_lower_gt_0": bool(ci[0] > 0)}
    return {"field": field, "model_name": sub["model_name"],
            "seed_mean_auc_pr": round(float(np.mean(g_ap)), 4),
            "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
            "delta_vs_M5prime": round(d, 4),
            "wilcoxon_p_raw": round(p_raw, 6), "p_BH": round(p_bh, 6),
            "bh_family_size": 1,
            "bh_note": ("one submission is a family of one, so p_BH equals the "
                        "raw p here. The paper's tables correct across the "
                        "architecture family within a discipline, which is "
                        "stricter; a submission compared against those numbers "
                        "should be corrected in the same family."),
            "student_ci95_vs_M5prime": ci,
            "gates": gates, "exceeds": all(gates.values())}


def demo_submission(d, field):
    """Build a submission from a directory of the paper's per-seed artifacts."""
    files = sorted(glob.glob(str(Path(d) / "*seed*.json")))
    per = {}
    for p in files:
        j = json.loads(Path(p).read_text())
        s = j.get("seed")
        if s in SEEDS and s not in per and "test_scores" in j:
            per[s] = j
    if sorted(per) != SEEDS:
        raise Reject(f"demo: {d} does not carry seeds {SEEDS} "
                     f"(found {sorted(per)})")
    _, pid, y = frozen_split(field)
    return {
        "benchmark": "research-area-persistence-v1",
        "field": field,
        "model_name": f"demo from {Path(d).name}",
        "seeds": SEEDS,
        "selection": {"split": "val", "metric": "auc_pr",
                      "note": "early stopping on the temporal validation cohort"},
        "test_student_pid": pid,
        "test_labels": [int(v) for v in y],
        "test_scores_per_seed": {str(s): [float(x) for x in per[s]["test_scores"]]
                                 for s in SEEDS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission")
    ap.add_argument("--demo-from")
    ap.add_argument("--field", default=None)
    ap.add_argument("--no-score", action="store_true",
                    help="validate only; do not compute the eq. (2) verdict")
    args = ap.parse_args()

    try:
        if args.demo_from:
            if not args.field:
                raise Reject("--demo-from needs --field")
            sub = demo_submission(args.demo_from, args.field)
            print(f"[validate] demo submission built from {args.demo_from}")
        elif args.submission:
            sub = json.loads(Path(args.submission).read_text())
        else:
            raise Reject("give --submission or --demo-from")

        ds, pid, y, checks = validate(sub, args.field)
        for c in checks:
            print(f"[validate] PASS check {c}")
        print(f"[validate] accepted: {sub['field']}, {len(pid)} test students, "
              f"{len(sub['seeds'])} seeds, selection on "
              f"{sub['selection']['split']}")
        if args.no_score:
            return 0
        r = score(sub, ds, y)
        print()
        print(f"[verdict] {r['model_name']}  mean AUC-PR "
              f"{r['seed_mean_auc_pr']:.4f}  M5' {r['M5prime_mean']:.4f}")
        print(f"[verdict] delta {r['delta_vs_M5prime']:+.4f}  CI "
              f"{r['student_ci95_vs_M5prime']}  p_BH {r['p_BH']:.4f}")
        print(f"[verdict] gates {r['gates']}  exceeds={r['exceeds']}")
        print(f"[verdict] p_BH corrects a family of one and so equals the raw "
              f"p; the paper's tables correct across the architecture family")
        print("[verdict] passing validation is not passing the gates")
        return 0
    except Reject as e:
        print(f"[validate] REJECTED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
