"""R1 - Label-threshold (theta) sensitivity sweep (P0-1, revision-robustness).

POST-HOC ROBUSTNESS CHECK, layered on top of the frozen artifact; NOT part of
the pre-registered protocol. The frozen tables are read-only throughout: the
label is rebuilt in memory as y_theta = 1[late_overlap > theta] for theta in
config.THETA_GRID, and everything downstream reruns with the pre-registered
0.05 decision thresholds unchanged.

Per (discipline, theta):
  1. base rates (overall and per temporal-split fold);
  2. the tabular ladder M0..M5 - e1_baselines.run imported verbatim (M4 comes
     along for free; the paper's theta table reports M0-M3 and M5);
  3. y-scrambling certificate - e9a_placebo.run_variant imported verbatim,
     30 seeds, global + within-cohort, e9a's pass rule re-evaluated;
  4. advisor-disjoint certificate - e9b_advisor_disjoint.run_split imported
     verbatim on the temporal and advisor-disjoint splits, e9b's -0.05 rule;
  5. advisor-placebo gap - e10_advisor_placebo.{placebo_overlap, run_variant}
     imported verbatim, L1 models only (the paper's mechanism gap is the
     true-minus-cohort-placebo L1 difference), 10 seeds, Wilcoxon + BH;
  6. student-only floor - mirrors e14_self_persistence.run_ladder with e14's
     fit helpers imported, then the pre-registered branch rule (e14 branch
     A/B/C thresholds 0.05, unchanged) with the placebo band recomputed at
     the same theta from step 5.

KNOWN DIVERGENCE FROM E14 (flagged, not silently fixed): e14's student
feature set includes early_concentration, a Herfindahl index computed from
the per-author works store, which is not present in this working copy (the
frozen tables ship without it). The sweep therefore runs the student-only
ladder on the four remaining student-side features (early_prod,
early_breadth, early_typicality, early_typicality_cohort) plus the student
TF-IDF arm. The theta=0.2 column of the output doubles as the calibration
against the frozen e14 floor (which included the fifth feature).

Split identity: temporal_split cuts on t0 quantiles only, so the folds are
byte-identical across theta; only the label vector moves.

Usage (one discipline per process; DATASET_PATH points at the frozen table):
  DATASET=chemistry DATASET_PATH=data/clean_dataset_chemistry.parquet \
      python code/paper_pipeline/experiments/r1_theta_sweep.py
  python code/paper_pipeline/experiments/r1_theta_sweep.py --merge

Outputs:
  results/robustness/theta_partial/<field>_theta_<t>.json   (per process)
  results/robustness/theta_<t>.json + theta_sweep_summary.json   (--merge)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

import e1_baselines as E1
import e9a_placebo as E9A
import e9b_advisor_disjoint as E9B
import e10_advisor_placebo as E10
from e14_self_persistence import (LEGACY_STUDENT_TFIDF, fit_gbdt, fit_logit,
                                  jaccard, paired_bootstrap_delta,
                                  student_tfidf, topk_set)
from collections import Counter

import pandas as pd
from scipy import sparse as sp

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "results" / "robustness"
PARTIAL_DIR = OUT_DIR / "theta_partial"
FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]
COHORT_WINDOW = 3  # e14.COHORT_WINDOW


def student_features_no_store(df):
    """e14_self_persistence.student_features minus early_concentration (the
    works-store Herfindahl); the four remaining columns are computed with the
    identical logic. Divergence documented in the module docstring."""
    early_sets = [set(l) for l in df.early_concepts]
    disc_top = topk_set(Counter(c for s in early_sets for c in s))
    typ = np.array([jaccard(s, disc_top) for s in early_sets])

    by_year = {}
    for s, t in zip(early_sets, df.t0.values):
        by_year.setdefault(int(t), Counter()).update(s)
    cohort_top = {}
    for t in sorted(by_year):
        pool = Counter()
        for u in range(t - COHORT_WINDOW, t + COHORT_WINDOW + 1):
            if u in by_year:
                pool.update(by_year[u])
        cohort_top[t] = topk_set(pool)
    typ_cohort = np.array([jaccard(s, cohort_top[int(t)])
                           for s, t in zip(early_sets, df.t0.values)])

    return pd.DataFrame({
        "early_prod": df.early_prod.astype(float).values,
        "early_breadth": df.early_breadth.astype(float).values,
        "early_typicality": typ,
        "early_typicality_cohort": typ_cohort,
    }, index=df.index)


def student_only_floor(df_split, X_st_tfidf, band):
    """Mirrors e14_self_persistence.run_ladder (source of every design and of
    the branch rule), on the reduced feature set, with the placebo band
    passed in (recomputed at the current theta) instead of read from the
    frozen e10 json."""
    F = student_features_no_store(df_split)
    masks = {k: (df_split.split == k).values for k in ("train", "val", "test")}
    y = {k: df_split.loc[m, "y"].values for k, m in masks.items()}

    Xt, cols = D.build_features(df_split, concepts="none")
    Xov = Xt[:, [cols.index("early_overlap")]]
    r_m1, s_m1 = fit_logit(Xov[masks["train"]], y["train"], Xov[masks["val"]],
                           y["val"], Xov[masks["test"]], y["test"], seed=0)

    designs = {
        "M1s_typicality": ("logit", F[["early_typicality"]].values, {}),
        "M1s_scalars": ("logit", F.values, {}),
        "M4s_tfidf": ("logit_sparse",
                      sp.hstack([sp.csr_matrix(F[["early_prod", "early_breadth"]].values),
                                 X_st_tfidf]).tocsr(), {"C_reg": 0.5}),
        "M3s_gbdt": ("gbdt",
                     np.hstack([F.values, X_st_tfidf.toarray()]).astype(np.float32), {}),
    }
    summary, test_scores = {}, {}
    for name, (kind, X, kw) in designs.items():
        runs = []
        for seed in C.SEEDS:
            a = (X[masks["train"]], y["train"], X[masks["val"]], y["val"],
                 X[masks["test"]], y["test"])
            if kind == "logit":
                r, s = fit_logit(*a, seed=seed)
            elif kind == "logit_sparse":
                r, s = fit_logit(*a, seed=seed, sparse=True, **kw)
            else:
                r, s = fit_gbdt(*a, seed=seed)
            runs.append(r)
            if seed == C.SEEDS[0]:
                test_scores[name] = s
        summary[name] = S.summarize_seeds(runs)

    m1_pr = r_m1["auc_pr"]
    best_name = max(designs, key=lambda n: summary[n]["auc_pr"]["mean"])
    best_pr = summary[best_name]["auc_pr"]["mean"]
    ci = paired_bootstrap_delta(y["test"], s_m1, test_scores[best_name])

    # e14's pre-registered branch rule, thresholds 0.05 unchanged
    band_hi = band + 0.05
    if best_pr <= band_hi and ci["ci95"][0] > 0:
        branch = "A_ADVISOR_INFORMATION_REQUIRED"
    elif (ci["ci95"][0] <= 0 <= ci["ci95"][1]) or best_pr >= m1_pr - 0.05:
        branch = "B_SELF_PERSISTENCE_EQUIVALENT"
    else:
        branch = "C_INTERMEDIATE"
    return {"summary": summary,
            "verdict": {"branch": branch, "best_student_model": best_name,
                        "best_student_auc_pr": round(best_pr, 4),
                        "M1_auc_pr": round(m1_pr, 4),
                        "placebo_band_plus_margin": round(band_hi, 4),
                        "M1_minus_floor_ci95": [round(v, 4) for v in ci["ci95"]]},
            "note": "student ladder WITHOUT early_concentration (works store absent)"}


def advisor_placebo_L1(df_split):
    """e10 restricted to the L1 rung (the paper's mechanism gap); the
    placebo generators and model code are e10's own."""
    variants = {"true": None, "placebo_cohort": "cohort",
                "placebo_random": "random", "field_mean": "field_mean"}
    per_variant = {}
    for name, kind in variants.items():
        runs = []
        for seed in C.SEEDS:
            ov = (df_split.early_overlap.values if kind is None
                  else E10.placebo_overlap(df_split, kind, seed))
            runs.append(E10.run_variant(df_split, ov, seed, thr_models=("L1",)))
        per_variant[name] = [r["L1_logit_overlap"]["auc_pr"] for r in runs]

    comparisons, pvals = [], []
    for v in ("placebo_cohort", "placebo_random", "field_mean"):
        t = S.paired_wilcoxon(per_variant["true"], per_variant[v])
        comparisons.append({"variant": v,
                            "true_mean": round(float(np.mean(per_variant["true"])), 4),
                            "placebo_mean": round(float(np.mean(per_variant[v])), 4),
                            "mean_diff": round(t["mean_diff"], 4), "p_raw": t["p"]})
        pvals.append(t["p"])
    p_adj, reject = S.bh_correction(pvals)
    for c, pa, rj in zip(comparisons, p_adj, reject):
        c["p_adj"] = round(float(pa), 6)
        c["significant"] = bool(rj)
    gap = comparisons[0]["true_mean"] - comparisons[0]["placebo_mean"]
    band = max(float(np.mean(per_variant[v]))
               for v in ("placebo_cohort", "placebo_random", "field_mean"))
    return {"L1_auc_pr_means": {v: round(float(np.mean(p)), 4)
                                for v, p in per_variant.items()},
            "gap_true_minus_cohort": round(gap, 4),
            "placebo_band_max_L1": round(band, 4),
            "comparisons": comparisons}


def run_theta(df, theta, X_st_tfidf):
    df = df.copy()
    df["y"] = (df.late_overlap > theta).astype(int)
    df_split = D.temporal_split(df)
    out = {"theta": theta,
           "base_rate": round(float(df.y.mean()), 4),
           "base_rate_by_split": {k: round(float(v), 4) for k, v in
                                  df_split.groupby("split").y.mean().items()}}

    # (2) ladder - e1.run verbatim, fresh cache per theta (NFA embeds y)
    cache, per_model = {}, {}
    for seed in C.SEEDS:
        for model, metrics in E1.run(seed, df_split, cache).items():
            per_model.setdefault(model, []).append(metrics)
    out["ladder"] = {m: S.summarize_seeds(v) for m, v in per_model.items()}
    best_tab = max(("M2_logit_tabular", "M3_gbdt_tabular"),
                   key=lambda m: out["ladder"][m]["auc_pr"]["mean"])
    out["best_pure_tabular"] = {"model": best_tab,
                                "auc_pr": out["ladder"][best_tab]["auc_pr"]["mean"]}

    # (3) y-scrambling - e9a verbatim
    X, _ = D.build_features(df_split, concepts="none")
    base = float(df_split.loc[df_split.split == "test", "y"].mean())
    scram = {}
    for name, fn in [("global", E9A.shuffle_global), ("cohort", E9A.shuffle_within_cohort)]:
        rows = E9A.run_variant(df_split, X, fn, list(range(E9A.N_PLACEBO_SEEDS)))
        rocs = np.array([r["auc_roc"] for r in rows])
        prs = np.array([r["auc_pr"] for r in rows])
        scram[name] = {"auc_roc_mean": round(float(rocs.mean()), 4),
                       "auc_roc_std": round(float(rocs.std(ddof=1)), 4),
                       "auc_pr_mean": round(float(prs.mean()), 4)}
    ok = (abs(scram["global"]["auc_roc_mean"] - 0.5) <= 0.035
          and abs(scram["cohort"]["auc_roc_mean"] - 0.5) <= 0.035
          and abs(scram["global"]["auc_pr_mean"] - base) <= 0.03)
    out["y_scrambling"] = {**scram, "test_base_rate": round(base, 4),
                           "verdict": "PASS" if ok else "FAIL"}

    # (4) advisor-disjoint - e9b verbatim
    ref, n_ref, br_ref = E9B.run_split(df_split, C.SEEDS)
    dis, n_dis, br_dis = E9B.run_split(D.advisor_disjoint_split(df, seed=0), C.SEEDS)
    shift = dis["auc_pr"]["mean"] - ref["auc_pr"]["mean"]
    out["advisor_disjoint"] = {
        "temporal_auc_pr": round(ref["auc_pr"]["mean"], 4),
        "disjoint_auc_pr": round(dis["auc_pr"]["mean"], 4),
        "shift_auc_pr": round(float(shift), 4),
        "test_base_rates": [round(br_ref, 4), round(br_dis, 4)],
        "verdict": "PASS" if shift >= -0.05 else "FAIL"}

    # (5) advisor-placebo gap - e10 L1
    out["advisor_placebo"] = advisor_placebo_L1(df_split)

    # (6) student-only floor + branch
    out["self_persistence"] = student_only_floor(
        df_split, X_st_tfidf, out["advisor_placebo"]["placebo_band_max_L1"])
    out["mechanism_branch"] = out["self_persistence"]["verdict"]["branch"]
    return out


def main_field():
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    df = D.load_dataset()
    # student tf-idf depends on early_concepts only: build once, reuse per theta
    df_probe = D.temporal_split(df.copy())
    X_st = student_tfidf(df_probe, legacy=LEGACY_STUDENT_TFIDF)
    for theta in C.THETA_GRID:
        p = PARTIAL_DIR / f"{field}_theta_{theta:.2f}.json"
        if p.exists():
            print(f"[r1] {field} theta={theta} already done, skipping", flush=True)
            continue
        print(f"[r1] {field} theta={theta}", flush=True)
        res = run_theta(df, theta, X_st)
        res["field"] = field
        p.write_text(json.dumps(res, indent=2))
        print(f"[r1] wrote {p} branch={res['mechanism_branch']}", flush=True)


def main_merge():
    grid = C.THETA_GRID
    summary = {"experiment": "R1_theta_sweep",
               "theta_grid": grid, "reference_theta": C.JACCARD_THETA,
               "note": ("post-hoc robustness; pre-registered 0.05 decision "
                        "thresholds unchanged; student floor without "
                        "early_concentration (works store absent)"),
               "fields": {}}
    for theta in grid:
        merged = {}
        for f in FIELDS:
            p = PARTIAL_DIR / f"{f}_theta_{theta:.2f}.json"
            merged[f] = json.loads(p.read_text())
        (OUT_DIR / f"theta_{theta:.2f}.json").write_text(json.dumps(merged, indent=2))
    for f in FIELDS:
        rows = {}
        for theta in grid:
            r = json.loads((PARTIAL_DIR / f"{f}_theta_{theta:.2f}.json").read_text())
            rows[f"{theta:.2f}"] = {
                "base_rate": r["base_rate"],
                "best_tabular_auc_pr": round(r["best_pure_tabular"]["auc_pr"], 4),
                "best_tabular_model": r["best_pure_tabular"]["model"],
                "M5_auc_pr": round(r["ladder"]["M5_gbdt_nfa"]["auc_pr"]["mean"], 4),
                "scramble": r["y_scrambling"]["verdict"],
                "disjoint_shift": r["advisor_disjoint"]["shift_auc_pr"],
                "disjoint": r["advisor_disjoint"]["verdict"],
                "placebo_gap": r["advisor_placebo"]["gap_true_minus_cohort"],
                "floor": r["self_persistence"]["verdict"]["best_student_auc_pr"],
                "branch": r["mechanism_branch"]}
        summary["fields"][f] = rows
    # branch-stability flags vs the pre-registered theta
    ref_key = f"{C.JACCARD_THETA:.2f}"
    for f, rows in summary["fields"].items():
        ref_branch = rows[ref_key]["branch"]
        summary["fields"][f] = {
            "rows": rows,
            "branch_stable_all_theta": all(v["branch"] == ref_branch for v in rows.values()),
            "branch_stable_adjacent": all(rows[k]["branch"] == ref_branch
                                          for k in ("0.15", "0.25")),
        }
    out = OUT_DIR / "theta_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        main_merge()
    else:
        main_field()
