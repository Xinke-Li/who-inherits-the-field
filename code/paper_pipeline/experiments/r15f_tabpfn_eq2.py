"""R15f - full eq. (2) protocol for the TabPFN screen-passers (B8).

TabPFN (CPU, 1000-row context) beat frozen M3 AND M5 per-seed in economics
and mathematics (the r15 screen). Post hoc rule (chemistry-RGCN precedent):
the corrected protocol decides, not the screen. This runs in the MAIN env,
consuming the per-row TabPFN test scores saved by the venv standalone
(results/robustness/extra_rungs_partial/<f>_tabpfn.json), aligning them to
the local temporal split by student_pid, refitting M5' identically to the
stored e12 run, and computing the paired student-level bootstrap of
TabPFN minus M5' (2000 draws, pooled over the 10 TabPFN seeds).

econ/math are the two smallest test cohorts (495, 935) and exactly the
disciplines the power analysis flags as not identifiable (MDE 0.07 to 0.10);
the student-level interval is expected to swallow the +0.02 to +0.03
seed-level gap, the same lesson as neuro CatBoost. Recorded either way.

Usage: DATASET=<econ|math> DATASET_PATH=... python r15f_tabpfn_eq2.py
Output: results/robustness/tabpfn_eq2_<field>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from e12_corrected_aggregation import N_BOOT, fast_auc_pr, fit_val_symmetric


def main():
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    rec = json.loads((REPO / "results" / "robustness" / "extra_rungs_partial"
                      / f"{field}_tabpfn.json").read_text())
    assert "test_scores_per_seed" in rec, f"{field}: no saved TabPFN scores"

    df_split = D.temporal_split(D.load_dataset())
    Xt, _ = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    p5 = D.split_xy(df_split, X5)
    (Xtr5, ytr), (Xva5, yva), (Xte5, yte) = p5["train"], p5["val"], p5["test"]
    te_pid_local = df_split.loc[df_split.split == "test", "student_pid"].astype(str).tolist()

    # align: the venv saved test rows in the same split order; assert by pid
    assert te_pid_local == rec["test_student_pid"], \
        f"{field}: TabPFN test rows do not align with the local split by pid"
    assert list(yte) == rec["test_labels"], f"{field}: label mismatch"

    tab_scores = {int(s): np.array(v, float)
                  for s, v in rec["test_scores_per_seed"].items()}
    tab_ap = [fast_auc_pr(yte, tab_scores[s]) for s in C.SEEDS]

    m5p_scores, m5p_ap = {}, []
    for s in C.SEEDS:
        p_te, _, it, _ = fit_val_symmetric(Xtr5, ytr, Xva5, yva, Xte5, s)
        m5p_scores[s] = p_te
        m5p_ap.append(fast_auc_pr(yte, p_te))
    stored = json.loads((REPO / "results" / f"results_{field}"
                         / "e12_corrected_vs_m5.json").read_text())
    m5p_stored = stored["ceilings"]["per_seed"]["M5_prime_val_symmetric"]
    dev = max(abs(a - b) for a, b in zip(m5p_ap, m5p_stored))
    assert dev < 5e-4, f"{field}: M5' refit deviates from stored e12 ({dev})"

    w = S.paired_wilcoxon(tab_ap, m5p_ap)
    rng = np.random.default_rng(0)
    n = len(yte)
    pooled = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = yte[idx]
        if yb.min() == yb.max():
            pooled[b] = np.nan
            continue
        pooled[b] = float(np.mean([fast_auc_pr(yb, tab_scores[s][idx])
                                   - fast_auc_pr(yb, m5p_scores[s][idx])
                                   for s in C.SEEDS]))
    pooled = pooled[~np.isnan(pooled)]
    ci = [round(float(np.percentile(pooled, 2.5)), 4),
          round(float(np.percentile(pooled, 97.5)), 4)]
    gates = {"seed_mean_gt_ceiling": bool(np.mean(tab_ap) > np.mean(m5p_ap)),
             "p_lt_0.05": bool(w["p"] < 0.05),
             "student_ci_lower_gt_0": bool(ci[0] > 0)}
    out = {"experiment": "R15f_tabpfn_eq2", "field": field,
           "model": rec["model"], "train_subsampled": rec["train_subsampled"],
           "tabpfn_mean": round(float(np.mean(tab_ap)), 4),
           "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
           "delta": round(float(np.mean(tab_ap) - np.mean(m5p_ap)), 4),
           "wilcoxon_p_raw": w["p"], "student_ci95": ci,
           "gates": gates, "exceeds": all(gates.values())}
    (REPO / "results" / "robustness" / f"tabpfn_eq2_{field}.json").write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
