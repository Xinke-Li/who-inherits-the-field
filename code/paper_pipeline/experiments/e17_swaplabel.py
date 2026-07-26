"""E17 - Swap-LABEL control (reviewer R1): remove the shared-label-side confound.

WHY. The label y = 1[J(C_late, A_true) > theta] is a function of the TRUE advisor
profile A_true. E10 swaps only the FEATURE side (early overlap vs a placebo
advisor) while holding y fixed, so part of E10's drop is a mechanical alignment
between a feature built from A_true and a label built from A_true, not an
inheritance dynamic. This control swaps BOTH sides onto a placebo advisor and
asks whether the true (student, advisor) pairing predicts better than a placebo
pairing, which cancels the shared-A_true advantage.

CONTROL. For each student, draw a cohort-matched placebo advisor A_p (|dt0|<=3,
different advisor). Build the placebo SYSTEM entirely from A_p:
    feature   eo_p = J(C_early_student, A_p)
    label     y_p  = 1[ J(C_late_student, A_p) > theta' ]
theta' is base-rate calibrated on TRAIN so mean(y_p) matches the frozen base
rate (so AUC-PR is comparable to the true system at the same base rate). Compare,
on the SAME temporal test split:
    true  system  AUC-PR( y_true , eo_true=early_overlap )   [ = M1, frozen ]
    placebo system AUC-PR( y_p    , eo_p )                    [ mechanical ]
The genuine advisor-specific signal is the EXCESS of the true system over the
swap-label placebo system; the raw E10 drop over-counts it.

PRE-REGISTERED DECISION RULE (fixed before running): the true (student, advisor)
pairing carries advisor-specific predictive structure beyond mechanical
label-feature sharing iff the true-system AUC-PR exceeds the swap-label placebo
system with a per-seed delta whose 95% interval excludes zero. theta' calibration
and 10 placebo redraws are the only stochastic parts; test is evaluated once.

Reuses e14.load_student_windows (C_late reconstruction, with the frozen
late_overlap self-check) and a cohort-matched placebo draw. Runs both disciplines.
Output: results_<disc>/e17_swaplabel.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S
import e14_self_persistence as E14

COHORT_WINDOW = 3


def calibrate_theta(lo_train, target_rate):
    """Smallest theta' on a fine grid whose train positive-rate is closest to
    the frozen base rate. Fixed rule, no test peeking."""
    grid = np.linspace(0.0, 0.9, 91)
    rates = np.array([(lo_train > th).mean() for th in grid])
    j = int(np.argmin(np.abs(rates - target_rate)))
    return float(grid[j]), float(rates[j])


def main():
    disc = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
    df = D.load_dataset()
    ds = D.temporal_split(df)
    base_rate = float(df.y.mean())

    early_cnt, late_set = E14.load_student_windows(df)
    early_sets = [set(l) for l in df.early_concepts]
    adv_profiles = [set(l) for l in df.adv_profile]
    late = [late_set[sid] for sid in df.st_openalex_id]

    # self-check: reconstructed J(C_late, A_true) reproduces frozen late_overlap
    recon = np.array([round(E14.jaccard(late[i], adv_profiles[i]), 4)
                      for i in range(len(df))])
    match = float(np.mean(np.abs(recon - df.late_overlap.values) < 1e-4))
    assert match > 0.98, f"late_overlap self-check only {match:.3f} - C_late reconstruction off"

    t0 = df.t0.values
    adv_pid = df.advisor_pid.values
    eo_true = df.early_overlap.values
    y_true = df.y.values
    is_test = (ds.split == "test").values
    is_train = (ds.split == "train").values

    # candidate placebo rows per student: cohort-matched, different advisor
    order = np.argsort(t0)
    per_seed = []
    for seed in C.SEEDS:
        rng = np.random.default_rng(seed)
        eo_p = np.empty(len(df)); lo_p = np.empty(len(df))
        # for each student pick a cohort-matched different-advisor row
        for i in range(len(df)):
            lo_i, hi_i = t0[i] - COHORT_WINDOW, t0[i] + COHORT_WINDOW
            cand = np.where((t0 >= lo_i) & (t0 <= hi_i) & (adv_pid != adv_pid[i]))[0]
            if len(cand) == 0:
                cand = np.where(adv_pid != adv_pid[i])[0]
            j = cand[rng.integers(len(cand))]
            eo_p[i] = E14.jaccard(early_sets[i], adv_profiles[j])
            lo_p[i] = E14.jaccard(late[i], adv_profiles[j])
        theta_p, tr_rate = calibrate_theta(lo_p[is_train], base_rate)
        y_p = (lo_p > theta_p).astype(int)
        placebo_auc = S.evaluate(y_p[is_test], eo_p[is_test])["auc_pr"]
        true_auc = S.evaluate(y_true[is_test], eo_true[is_test])["auc_pr"]
        per_seed.append({"seed": seed, "theta_p": round(theta_p, 3),
                         "placebo_base_rate_test": round(float(y_p[is_test].mean()), 4),
                         "swaplabel_placebo_auc_pr": round(placebo_auc, 4),
                         "true_system_auc_pr": round(true_auc, 4),
                         "excess_true_minus_placebo": round(true_auc - placebo_auc, 4)})
        print(f"[e17:{disc}] seed {seed} theta'={theta_p:.2f} "
              f"placebo={placebo_auc:.3f} true={true_auc:.3f} "
              f"excess={true_auc - placebo_auc:+.3f}", flush=True)

    pl = np.array([r["swaplabel_placebo_auc_pr"] for r in per_seed])
    tr = np.array([r["true_system_auc_pr"] for r in per_seed])
    ex = tr - pl
    # per-seed 95% interval on the excess (percentile over placebo redraws)
    ci = [round(float(np.percentile(ex, 2.5)), 4), round(float(np.percentile(ex, 97.5)), 4)]
    out = {
        "experiment": "E17_swaplabel_control", "discipline": disc,
        "base_rate": round(base_rate, 4), "late_overlap_selfcheck_match": round(match, 4),
        "true_system_auc_pr": round(float(tr.mean()), 4),
        "swaplabel_placebo_auc_pr_mean": round(float(pl.mean()), 4),
        "swaplabel_placebo_auc_pr_std": round(float(pl.std()), 4),
        "excess_true_minus_placebo_mean": round(float(ex.mean()), 4),
        "excess_ci95": ci,
        "excess_excludes_zero": bool(ci[0] > 0),
        "e10_raw_placebo_auc_pr_reference": 0.158 if disc == "econ" else 0.276,
        "per_seed": per_seed,
        "reading": ("advisor-specific structure survives the swap-label control: the "
                    "true pairing out-predicts a placebo pairing whose label and "
                    "feature share the same placebo advisor, so the E10 drop is not "
                    "merely mechanical label-feature sharing"
                    if ci[0] > 0 else
                    "the true pairing does NOT out-predict the swap-label placebo: "
                    "the E10 drop is largely mechanical; report accordingly"),
    }
    (C.RESULTS_DIR / "e17_swaplabel.json").write_text(json.dumps(out, indent=2))
    print(f"[e17:{disc}] true {out['true_system_auc_pr']} vs swaplabel-placebo "
          f"{out['swaplabel_placebo_auc_pr_mean']} | excess {out['excess_true_minus_placebo_mean']} "
          f"CI {ci} excludes0={out['excess_excludes_zero']}")


if __name__ == "__main__":
    main()
