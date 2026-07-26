#!/usr/bin/env python3
"""T2.2a, continued: TOST for the genealogy arm, and the coverage question asked
in a way that is not confounded with training-set size.

Three analyses, all reported side by side.

FULL          train on all rows, evaluate on all test rows. The primary result.
COVERED-EVAL  train on all rows exactly as above, evaluate only on the test rows
              that actually have a grand-advisor with cached works. This isolates
              coverage from sample size and asks the intended question: among
              students whose genealogy is observed, does the model that can use
              it do better? This is the primary covered analysis.
COVERED-RETRAIN
              train and evaluate on covered rows only. Reported as secondary and
              explicitly confounded: it shrinks the training set as well, so it
              answers "does genealogy help in a much smaller training regime".

Every interval is reported next to the n that produced it, and both counts are
given: covered modelling rows and covered TEST rows. The test count is what
governs the intervals.

TOST uses r11_power_tost.py's machinery at the identical margins.

  python code/r24_genealogy_tost.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
OUT = ROOT / "results" / "revision" / "T2_2a_genealogy_tabular"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
SEEDS = list(range(10))
N_BOOT = 2000
MARGINS = [0.01, 0.02, 0.03]

from r23_genealogy_tabular import (FROZEN_M5, GFEATS, build_genealogy,  # noqa: E402
                                   fast_auc_pr)


def tost(mean, lo, hi, exceeds):
    sigma = (hi - lo) / 3.92
    lo90, hi90 = mean - 1.645 * sigma, mean + 1.645 * sigma
    eq_at = next((m for m in MARGINS if -m < lo90 and hi90 < m), None)
    verdict = ("exceeds" if exceeds else
               f"equivalent at {eq_at}" if eq_at is not None else
               "not identifiable at this design")
    return {"delta": round(mean, 4), "sigma": round(sigma, 4),
            "MDE_80pct": round(2.80 * sigma, 4),
            "ci90": [round(lo90, 4), round(hi90, 4)],
            "equivalent_at_margin": eq_at, "verdict": verdict,
            "largest_gain_excluded": round(hi90, 4)}


def gate2(y, sc_m5, sc_m6, ap5, ap6, S, tag, n_model, n_test):
    """eq. (2) on one evaluation set. sc_* are per-seed score arrays on that set."""
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    b5 = np.full((len(SEEDS), N_BOOT), np.nan)
    b6 = np.full((len(SEEDS), N_BOOT), np.nan)
    for b in range(N_BOOT):
        i = idx[b]
        yb = y[i]
        if yb.sum() in (0, len(yb)):
            continue
        for s in range(len(SEEDS)):
            b5[s, b] = fast_auc_pr(yb, sc_m5[s][i])
            b6[s, b] = fast_auc_pr(yb, sc_m6[s][i])
    d = np.array(ap6) - np.array(ap5)
    praw = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    padj = float(S.bh_correction([praw])[0][0])
    pooled = np.nanmean(b6 - b5, axis=0)
    ci = [float(np.nanpercentile(pooled, 2.5)), float(np.nanpercentile(pooled, 97.5))]
    dbar = float(np.mean(ap6) - np.mean(ap5))
    gates = {"mean_gt_ceiling": bool(dbar > 0), "p_adj_lt_0.05": bool(padj < 0.05),
             "student_ci_lower_gt_0": bool(ci[0] > 0)}
    exceeds = all(gates.values())
    return {"analysis": tag,
            "n_modelling_rows": int(n_model), "n_test_rows": int(n_test),
            "n_test_positives": int(y.sum()),
            "M5_mean_auc_pr": round(float(np.mean(ap5)), 4),
            "M6_mean_auc_pr": round(float(np.mean(ap6)), 4),
            "delta_vs_M5": round(dbar, 4), "p_BH": round(padj, 6),
            "student_ci95": [round(c, 4) for c in ci],
            "gates": gates, "exceeds": exceeds,
            "tost": tost(dbar, ci[0], ci[1], exceeds)}


def fit_arms(dsp, D):
    """Fit M5 and M6 on dsp's own train split; return test y and per-seed scores."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xt, _ = D.build_features(dsp, concepts="none")
    nfa = D.build_nfa_features(dsp)
    base = np.hstack([Xt, nfa.values.astype(float)])
    arms = {"M5": base,
            "M6": np.hstack([base, dsp[GFEATS].values.astype(float)])}
    out = {}
    for name, X in arms.items():
        sp = D.split_xy(dsp, X)
        (Xtr, ytr), _, (Xte, yte) = sp["train"], sp["val"], sp["test"]
        ss = []
        for seed in SEEDS:
            m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                               early_stopping=True,
                                               validation_fraction=0.15)
            m.fit(Xtr, ytr)
            ss.append(m.predict_proba(Xte)[:, 1])
        out[name] = ss
        out["y_test"] = yte
    return out


def run_one(field):
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
    from utils import data as D
    from utils import stats as S

    dsp = D.temporal_split(D.load_dataset())
    g, cov, withworks = build_genealogy(field, dsp)
    for c in GFEATS + ["has_grandadv_works"]:
        dsp[c] = g[c].values

    # ---- FULL: train on all, evaluate on all test ----
    f = fit_arms(dsp, D)
    y = f["y_test"]
    ap5 = [fast_auc_pr(y, s) for s in f["M5"]]
    ap6 = [fast_auc_pr(y, s) for s in f["M6"]]
    full = gate2(y, f["M5"], f["M6"], ap5, ap6, S, "full",
                 len(dsp), len(y))
    full["gate1_frozen_M5"] = FROZEN_M5[field]
    full["gate1_delta"] = round(full["M5_mean_auc_pr"] - FROZEN_M5[field], 4)
    full["gate1_matches_3dp"] = abs(full["gate1_delta"]) < 5e-4

    # ---- COVERED-EVAL: same fitted models, evaluate on covered test rows only ----
    te_mask = (dsp.split == "test").values
    cov_test = dsp.loc[te_mask, "has_grandadv_works"].values.astype(bool)
    ce = None
    if cov_test.sum() >= 30 and 0 < y[cov_test].sum() < cov_test.sum():
        yc = y[cov_test]
        s5 = [s[cov_test] for s in f["M5"]]
        s6 = [s[cov_test] for s in f["M6"]]
        ce = gate2(yc, s5, s6, [fast_auc_pr(yc, s) for s in s5],
                   [fast_auc_pr(yc, s) for s in s6], S, "covered_eval",
                   len(dsp), len(yc))
        ce["note"] = ("trained on all rows exactly as the full arm; evaluation "
                      "restricted to covered test rows, so training size is held "
                      "fixed and only coverage varies")

    # ---- COVERED-RETRAIN: secondary, confounded with training size ----
    sub = dsp[dsp.has_grandadv_works == 1].copy()
    cr = None
    if sub.split.nunique() == 3 and sub.groupby("split").y.nunique().min() > 1:
        fr = fit_arms(sub, D)
        yr = fr["y_test"]
        if 0 < yr.sum() < len(yr):
            cr = gate2(yr, fr["M5"], fr["M6"],
                       [fast_auc_pr(yr, s) for s in fr["M5"]],
                       [fast_auc_pr(yr, s) for s in fr["M6"]],
                       S, "covered_retrain", len(sub), len(yr))
            cr["note"] = ("CONFOUNDED: the training set shrinks with the "
                          "evaluation set, so this answers whether genealogy "
                          "helps in a smaller training regime, not whether it "
                          "helps where genealogy is observed")

    out = {"field": field,
           "grand_advisor_coverage": round(cov, 4),
           "grand_advisor_with_cached_works": round(withworks, 4),
           "n_test_covered": int(cov_test.sum()),
           "full": full, "covered_eval": ce, "covered_retrain": cr}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"tost_{field}.json", "w"), indent=2)

    print(f"[FULL] {field:10} testN {full['n_test_rows']:>5} "
          f"M5 {full['M5_mean_auc_pr']:.4f} (frozen {FROZEN_M5[field]:.3f}, "
          f"d{full['gate1_delta']:+.4f})  M6 d{full['delta_vs_M5']:+.4f} "
          f"CI {full['student_ci95']} -> {full['tost']['verdict']}")
    if ce:
        print(f"[CEVL] {field:10} testN {ce['n_test_rows']:>5} "
              f"pos {ce['n_test_positives']:>4}  M5 {ce['M5_mean_auc_pr']:.4f} "
              f"M6 d{ce['delta_vs_M5']:+.4f} CI {ce['student_ci95']} "
              f"MDE {ce['tost']['MDE_80pct']:.4f} -> {ce['tost']['verdict']}")
    if cr:
        print(f"[CRTR] {field:10} testN {cr['n_test_rows']:>5} "
              f"pos {cr['n_test_positives']:>4}  M5 {cr['M5_mean_auc_pr']:.4f} "
              f"M6 d{cr['delta_vs_M5']:+.4f} CI {cr['student_ci95']} "
              f"MDE {cr['tost']['MDE_80pct']:.4f} -> {cr['tost']['verdict']}")
    return 0


def main():
    if os.environ.get("R24_FIELD"):
        return run_one(os.environ["R24_FIELD"])
    OUT.mkdir(parents=True, exist_ok=True)
    for fl in FIELDS:
        env = dict(os.environ)
        env.update(R24_FIELD=fl, DATASET=fl,
                   DATASET_PATH=str(ROOT / "data" / f"clean_dataset_{fl}.parquet"))
        env.pop("NEURO_DATASET", None)
        subprocess.call([sys.executable, str(Path(__file__).resolve())], env=env)

    per = {fl: json.load(open(OUT / f"tost_{fl}.json")) for fl in FIELDS
           if (OUT / f"tost_{fl}.json").exists()}
    summary = {"task": "T2.2a TOST and coverage control", "margins": MARGINS,
               "machinery": "identical to r11_power_tost.py",
               "primary_covered_analysis": "covered_eval (training size held fixed)",
               "fields": per,
               "any_exceeds": {
                   k: any((r.get(k) or {}).get("exceeds", False) for r in per.values())
                   for k in ("full", "covered_eval", "covered_retrain")}}
    json.dump(summary, open(OUT / "tost_summary.json", "w"), indent=2)
    print(f"\n[T2.2a] any exceeds: {summary['any_exceeds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
