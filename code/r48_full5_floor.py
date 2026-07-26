#!/usr/bin/env python3
"""T2.4 step 2 - the student-only floor at FIVE features, over both sweeps.

WHAT CHANGES AND WHAT DOES NOT. r1_theta_sweep.student_only_floor is the source
of every design, every seed, and the branch rule; the only edit is the feature
frame, which regains early_concentration and so has five columns instead of the
four r1 could build (r1_theta_sweep.py:26-32). The pre-registered 0.05 branch
thresholds are unchanged, the temporal split is unchanged, the TF-IDF arm is
unchanged, and the paired bootstrap is e14's, 2000 draws at seed 0.

THE TF-IDF FOOTING, found the hard way and worth stating first. Every recorded
student-only floor, the frozen e14 certificate and both sweeps alike, was
produced when student_tfidf fitted the vectoriser on ALL rows. The W5 audit
repair changed the default to fit on the training split only, and the change was
never carried back through the recorded artifacts. Rerunning r1's own
four-feature floor today therefore does NOT reproduce r1's own recorded numbers:
mathematics at theta 0.2 gives M4s_tfidf 0.2483 against the recorded 0.2519, and
student_tfidf(..., legacy=True) returns the recorded value exactly. The repair
moves more than the fifth feature does, so mixing the two footings in one
comparison would attribute the repair's effect to the feature.

This script therefore runs BOTH arms and keeps them apart:
  --tfidf legacy    (default) the footing of every recorded number, so the five
                    feature floor is comparable to the recorded four-feature
                    sweep and to the frozen e14 certificate. Partials land in
                    partial/.
  --tfidf repaired  the current default, the W5 train-only fit. Partials land in
                    partial_w5repaired/, beside the four-feature control that
                    was run on the same footing.
Neither arm is deleted and neither is presented alone.

WHAT IS READ RATHER THAN RECOMPUTED. Each cell's placebo band comes from the
matching cell of the existing sweep, results/robustness/theta_<t>.json and
topk_partial/<field>_k<k>.json. The band is the e10 L1 rung on advisor-side
placebos and contains no student-side feature, so restoring the fifth student
feature cannot move it. Recomputing it would burn four variants at ten seeds
per cell to reproduce a number the artifact already carries.

PROVENANCE. early_concentration is r47's reconstruction from the r5 live-API
cache, not the frozen builder store, which is absent. Everything downstream of
it inherits that. The calibration block written at theta 0.2 / k 10 reports the
five-feature floor against the FROZEN e14 floor per field, which is the only
measurement that bounds the drift; read the sweep against that, not against
frozen numbers directly.

Outputs
  results/revision/T2_4_e14_full5/partial/<field>_theta_<t>.json
  results/revision/T2_4_e14_full5/partial/<field>_k<k>.json
  results/revision/T2_4_e14_full5/partial/<field>_k10_ms<ms>.json
  results/revision/T2_4_e14_full5/full5_floor_summary.json          (--merge)

Usage
  DATASET=math DATASET_PATH=data/clean_dataset_math.parquet \
      python code/r48_full5_floor.py --sweep theta
  python code/r48_full5_floor.py --merge
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

import config as C                                              # noqa: E402
from utils import data as D                                     # noqa: E402
from utils import stats as S                                    # noqa: E402
from e14_self_persistence import (fit_gbdt, fit_logit,           # noqa: E402
                                  paired_bootstrap_delta, student_tfidf)
from r1_theta_sweep import student_features_no_store             # noqa: E402
from r6_topk_sweep import (K_GRID, K_REF, MIN_SCORE_GRID,        # noqa: E402
                           BUILDER_MIN_SCORE, apply_rebuild,
                           load_cache_for, rebuild)

OUT = ROOT / "results" / "revision" / "T2_4_e14_full5"
PARTIAL_BY_MODE = {"legacy": OUT / "partial",
                   "repaired": OUT / "partial_w5repaired"}
PARTIAL = PARTIAL_BY_MODE["legacy"]
TFIDF_MODE = "legacy"
SUPP = ROOT / "data" / "supplement"
ROB = ROOT / "results" / "robustness"
FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]


def concentration_map(field, min_score):
    """student_pid -> early_concentration at the requested concept min-score.
    The builder's 0.3 comes from the supplement table T2.4 specifies; the two
    variants come from r47's side file, which the supplement deliberately omits."""
    if min_score == BUILDER_MIN_SCORE:
        df = pd.read_parquet(SUPP / f"early_concentration_{field}.parquet")
        col = "early_concentration"
    else:
        df = pd.read_parquet(OUT / f"early_concentration_variants_{field}.parquet")
        col = f"early_concentration_ms{min_score}"
    return dict(zip(df.student_pid, df[col]))


def student_features_full5(df, conc):
    """r1's four columns, verbatim from r1, plus the fifth in e14's position.

    e14_self_persistence.student_features orders the frame
    early_prod, early_breadth, early_typicality, early_typicality_cohort,
    early_concentration; the same order is kept here so the two frames differ
    only by the presence of the last column."""
    F = student_features_no_store(df)
    F["early_concentration"] = [conc[p] for p in df.student_pid]
    return F


def floor_full5(df_split, X_st_tfidf, band, conc):
    """r1_theta_sweep.student_only_floor with the five-column frame.

    Every other line is r1's: the same four designs, the same ten seeds, the
    same C_reg on the sparse rung, the same deterministic M1 reference, the
    same 2000-draw paired bootstrap, the same 0.05 thresholds."""
    F = student_features_full5(df_split, conc)
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
                        "M1_minus_floor_ci95": [round(v, 4) for v in ci["ci95"]],
                        "M1_minus_floor_ci95_full": ci["ci95"],
                        "M1_minus_floor_excludes_zero": ci["excludes_zero"]},
            "features": list(student_features_full5(df_split, conc).columns),
            "note": "student ladder WITH early_concentration (r47 reconstruction)"}


def run_theta_sweep(field, df, only=None):
    """--only runs one theta cell, named as its partial file's stem."""
    thetas = [t for t in C.THETA_GRID
              if only is None or f"{field}_theta_{t:.2f}" == only]
    if only is not None and not thetas:
        return
    df_probe = D.temporal_split(df.copy())
    X_st = student_tfidf(df_probe, legacy=(TFIDF_MODE == "legacy"))
    conc = concentration_map(field, BUILDER_MIN_SCORE)
    for theta in thetas:
        p = PARTIAL / f"{field}_theta_{theta:.2f}.json"
        if p.exists():
            print(f"[r48] {field} theta={theta} done, skipping", flush=True)
            continue
        band = json.loads((ROB / f"theta_{theta:.2f}.json").read_text())[field][
            "advisor_placebo"]["placebo_band_max_L1"]
        d = df.copy()
        d["y"] = (d.late_overlap > theta).astype(int)
        res = floor_full5(D.temporal_split(d), X_st, band, conc)
        res.update({"field": field, "sweep": "theta", "theta": theta,
                    "tfidf_fit": TFIDF_MODE,
                    "band_source": f"results/robustness/theta_{theta:.2f}.json",
                    "placebo_band_max_L1": band})
        p.write_text(json.dumps(res, indent=2))
        print(f"[r48] {field} theta={theta}: floor "
              f"{res['verdict']['best_student_auc_pr']} ci "
              f"{res['verdict']['M1_minus_floor_ci95']} branch "
              f"{res['verdict']['branch']}", flush=True)


def run_topk_sweep(field, df, only=None):
    """One process can do the whole grid or exactly one cell (--only).

    Single-cell mode exists for chemistry, whose 33,528-author slice of the r5
    cache and whose 26,830-row dense TF-IDF matrix do not comfortably coexist on
    a 16 GB machine. The cache is dropped as soon as the rebuild has consumed
    it, before any model is fitted, and the process exits after the cell, so
    nothing is carried into the next one. The numbers are unaffected: the cell
    sees the identical rebuilt table either way."""
    cells = [(k, BUILDER_MIN_SCORE, f"{field}_k{k}") for k in K_GRID]
    cells += [(K_REF, ms, f"{field}_k{K_REF}_ms{ms}") for ms in MIN_SCORE_GRID]
    if only:
        cells = [c for c in cells if c[2] == only]
        if not cells:
            raise SystemExit(f"r48: --only {only} matches no cell of {field}")
    todo = [c for c in cells if not (PARTIAL / f"{c[2]}.json").exists()]
    for _, _, stem in cells:
        if (PARTIAL / f"{stem}.json").exists():
            print(f"[r48] {stem} done, skipping", flush=True)
    if not todo:
        return
    aids = set(df.st_openalex_id) | set(df.adv_openalex_id)
    print(f"[r48] {field}: loading r5 cache for {len(aids)} authors", flush=True)
    cache = load_cache_for(aids)
    for i, (k, ms, stem) in enumerate(todo):
        p = PARTIAL / f"{stem}.json"
        band = json.loads((ROB / "topk_partial" / f"{stem}.json").read_text())[
            "advisor_placebo"]["placebo_band_max_L1"]
        dfk, drops = apply_rebuild(df, rebuild(df, cache, k, ms))
        if i == len(todo) - 1:
            del cache
            gc.collect()
            print(f"[r48] r5 cache released before fitting {stem}", flush=True)
        df_split = D.temporal_split(dfk)
        X_st = student_tfidf(df_split, legacy=(TFIDF_MODE == "legacy"))
        res = floor_full5(df_split, X_st, band, concentration_map(field, ms))
        res.update({"field": field, "sweep": "topk", "k": k, "min_score": ms,
                    "tfidf_fit": TFIDF_MODE,
                    "band_source": f"results/robustness/topk_partial/{stem}.json",
                    "placebo_band_max_L1": band, "row_diagnostics": drops,
                    "provenance": "rebuilt from the r5 live-API cache; compare "
                                  "within the rebuilt family only"})
        p.write_text(json.dumps(res, indent=2))
        print(f"[r48] {stem}: floor {res['verdict']['best_student_auc_pr']} ci "
              f"{res['verdict']['M1_minus_floor_ci95']} branch "
              f"{res['verdict']['branch']}", flush=True)


def merge():
    frozen_label = {"A_ADVISOR_INFORMATION_REQUIRED": "advisor-required",
                    "B_SELF_PERSISTENCE_EQUIVALENT": "self-persistence",
                    "C_INTERMEDIATE": "advisor-adds"}
    out = {"task": "T2.4",
           "what": ("the student-only floor of r1_theta_sweep and r6_topk_sweep, "
                    "rerun with all five e14 student-side features"),
           "fifth_feature_source": ("r47_early_concentration.py, reconstructed "
                                    "from results/robustness/openalex_cache/ "
                                    "(r5 live-API fetch), because the builder's "
                                    "works store is in neither repository nor "
                                    "archive"),
           "branch_rule": "e14 branch A/B/C, thresholds 0.05, unchanged",
           "branch_label_map": frozen_label,
           "tfidf_footing": {
               "primary": "legacy (all-rows fit), the footing of every recorded "
                          "number, so four against five isolates the feature",
               "second_arm": "partial_w5repaired/, the current train-only fit",
               "why": ("the W5 repair changed student_tfidf's default and was "
                       "never carried back through the recorded artifacts; "
                       "r1's own four-feature floor does not reproduce r1's "
                       "recorded numbers under the repaired default")},
           "fields": {}}
    for f in FIELDS:
        rows = {}
        for theta in C.THETA_GRID:
            p = PARTIAL / f"{f}_theta_{theta:.2f}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            old = json.loads((ROB / f"theta_{theta:.2f}.json").read_text())[f][
                "self_persistence"]["verdict"]
            rows[f"theta_{theta:.2f}"] = _row(r, old)
        for k in K_GRID:
            p = PARTIAL / f"{f}_k{k}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            old = json.loads((ROB / "topk_partial" / f"{f}_k{k}.json").read_text())[
                "self_persistence"]["verdict"]
            rows[f"k{k}"] = _row(r, old)
        for ms in MIN_SCORE_GRID:
            p = PARTIAL / f"{f}_k{K_REF}_ms{ms}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            old = json.loads((ROB / "topk_partial" /
                              f"{f}_k{K_REF}_ms{ms}.json").read_text())[
                "self_persistence"]["verdict"]
            rows[f"k{K_REF}_ms{ms}"] = _row(r, old)
        rep = {}
        for stem, key in ([(f"{f}_theta_{t:.2f}", f"theta_{t:.2f}")
                           for t in C.THETA_GRID]
                          + [(f"{f}_k{k}", f"k{k}") for k in K_GRID]
                          + [(f"{f}_k{K_REF}_ms{ms}", f"k{K_REF}_ms{ms}")
                             for ms in MIN_SCORE_GRID]):
            q = PARTIAL_BY_MODE["repaired"] / f"{stem}.json"
            if q.exists():
                v = json.loads(q.read_text())["verdict"]
                rep[key] = {"floor": v["best_student_auc_pr"],
                            "ci": v["M1_minus_floor_ci95"],
                            "branch": v["branch"]}
        out["fields"][f] = {
            "cells": rows,
            "n_cells": len(rows),
            "n_branch_moves": sum(1 for v in rows.values()
                                  if v["branch_four"] != v["branch_five"]),
            "five_feature_w5repaired_arm": rep}

    # the frozen e14 calibration: five features against the frozen certificate
    calib = {}
    for f in FIELDS:
        p = PARTIAL / f"{f}_theta_0.20.json"
        fz = ROOT / "results" / f"results_{f}" / "e14_self_persistence.json"
        if not (p.exists() and fz.exists()):
            continue
        five = json.loads(p.read_text())["verdict"]
        frozen = json.loads(fz.read_text())["a_student_only_ladder"]["verdict"]
        frozen_ci = json.loads(fz.read_text())["a_student_only_ladder"][
            "comparisons"][frozen["best_student_model"]]["vs_M1_bootstrap"]["ci95"]
        calib[f] = {
            "frozen_e14_floor": frozen["best_student_auc_pr"],
            "five_feature_floor": five["best_student_auc_pr"],
            "floor_delta": round(five["best_student_auc_pr"]
                                 - frozen["best_student_auc_pr"], 4),
            "frozen_e14_rung": frozen["best_student_model"],
            "five_feature_rung": five["best_student_model"],
            "frozen_e14_ci95": [round(v, 4) for v in frozen_ci],
            "five_feature_ci95": five["M1_minus_floor_ci95"],
            "frozen_e14_branch": frozen["branch"],
            "five_feature_branch": five["branch"]}
    out["calibration_vs_frozen_e14"] = calib
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "full5_floor_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"calibration_vs_frozen_e14": calib}, indent=2))
    print("[r48] ->", OUT / "full5_floor_summary.json")


def _row(r, old):
    v = r["verdict"]
    return {"floor_four": old["best_student_auc_pr"],
            "floor_five": v["best_student_auc_pr"],
            "floor_delta": round(v["best_student_auc_pr"]
                                 - old["best_student_auc_pr"], 4),
            "rung_four": old["best_student_model"],
            "rung_five": v["best_student_model"],
            "M1": v["M1_auc_pr"],
            "ci_four": old["M1_minus_floor_ci95"],
            "ci_five": v["M1_minus_floor_ci95"],
            "branch_four": old["branch"],
            "branch_five": v["branch"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["theta", "topk", "both"], default="both")
    ap.add_argument("--tfidf", choices=["legacy", "repaired"], default="legacy")
    ap.add_argument("--only", default=None,
                    help="run exactly one topk cell, e.g. chemistry_k10")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    TFIDF_MODE = args.tfidf
    PARTIAL = PARTIAL_BY_MODE[TFIDF_MODE]
    PARTIAL.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge()
    else:
        field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
        df = D.load_dataset()
        if args.sweep in ("theta", "both"):
            run_theta_sweep(field, df, only=args.only)
        if args.sweep in ("topk", "both"):
            run_topk_sweep(field, df, only=args.only)
