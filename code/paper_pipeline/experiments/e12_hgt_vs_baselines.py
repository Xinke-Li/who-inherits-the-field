"""E12 - Head-to-head significance test: HGT vs the tabular ceiling (added 2026-07-05).

Fixes a reviewer-facing gap: E3 tested the HGT *ablations* with paired Wilcoxon
+ BH, but the paper's headline comparison (HGT vs M5 / M2 / M3) carried no
test. This script computes, from the existing per-seed artifacts:

  (i)  paired per-seed Wilcoxon, HGT (ablate=none) vs M5 / M3 / M2
       (seeds 0-9 pair naturally: same split, same seed grid), BH-corrected;
  (ii) a score-level paired bootstrap on the test set for seed 0
       (HGT scores from results_hgt/hgt_none_seed0.json vs an M5 refit),
       reporting the CI of the AUC-PR difference.

Inputs : results_econ/results_hgt/hgt_none_seed*.json, results_econ/e1_baselines.json
Output : results_econ/e12_hgt_vs_baselines.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S


def load_hgt_per_seed(variant="standard"):
    """variant='standard' -> results_hgt/hgt_none_seed*.json
       variant='tuned'    -> results_hgt_grid/hgt_none_seed*_tuned.json"""
    if variant == "tuned":
        files = sorted((C.RESULTS_DIR / "results_hgt_grid").glob("hgt_none_seed*_tuned.json"))
    else:
        files = sorted((C.RESULTS_DIR / "results_hgt").glob("hgt_none_seed*.json"))
    runs = {}
    for f in files:
        d = json.loads(f.read_text())
        runs[int(d["seed"])] = d
    assert set(runs) >= set(C.SEEDS), f"missing HGT seeds ({variant}): {set(C.SEEDS) - set(runs)}"
    return runs


def load_e1_per_seed():
    """Per-seed baseline metrics from e1_baselines.json (per_seed lists are
    ordered by C.SEEDS; e1_perseed.jsonl predates M5 and is not used)."""
    d = json.loads((C.RESULTS_DIR / "e1_baselines.json").read_text())
    per = d["per_seed"]
    return {s: {m: per[m][i] for m in per} for i, s in enumerate(d["seeds"])}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="standard", choices=["standard", "tuned"])
    args = ap.parse_args()

    hgt = load_hgt_per_seed(args.variant)
    e1 = load_e1_per_seed()

    hgt_pr = [hgt[s]["auc_pr"] for s in C.SEEDS]
    out = {"experiment": "E12_hgt_vs_baselines", "variant": args.variant,
           "hgt_hp": hgt[0].get("hp"), "seeds": C.SEEDS,
           "hgt_auc_pr": {"mean": round(float(np.mean(hgt_pr)), 4),
                          "std": round(float(np.std(hgt_pr, ddof=1)), 4)},
           "tests": {}}

    comparisons, pvals = [], []
    for base in ("M5_gbdt_nfa", "M3_gbdt_tabular", "M2_logit_tabular"):
        b = [e1[s][base]["auc_pr"] for s in C.SEEDS]
        t = S.paired_wilcoxon(hgt_pr, b)
        comparisons.append({"baseline": base,
                            "baseline_mean": round(float(np.mean(b)), 4),
                            "hgt_mean": round(float(np.mean(hgt_pr)), 4),
                            "mean_diff_hgt_minus_base": round(t["mean_diff"], 4),
                            "p_raw": t["p"]})
        pvals.append(t["p"])
    p_adj, reject = S.bh_correction(pvals)
    for c, pa, rj in zip(comparisons, p_adj, reject):
        c["p_adj"] = round(float(pa), 6)
        c["significant"] = bool(rj)
    out["tests"]["paired_wilcoxon_bh"] = comparisons

    # (ii) score-level paired bootstrap, seed 0: HGT vs M5 refit on identical split
    d0 = hgt[0]
    y_te = np.array(d0["test_labels"])
    p_hgt = np.array(d0["test_scores"])

    df = D.load_dataset()
    df_split = D.temporal_split(df)
    Xt, _ = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    parts = D.split_xy(df_split, X5)
    (Xtr, ytr), _, (Xte, yte) = parts["train"], parts["val"], parts["test"]
    assert len(yte) == len(y_te) and (yte == y_te).all(), \
        "HGT and local test splits disagree - check split bounds"
    from sklearn.ensemble import HistGradientBoostingClassifier
    m5 = HistGradientBoostingClassifier(random_state=0, max_iter=500,
                                        early_stopping=True, validation_fraction=0.15)
    m5.fit(Xtr, ytr)
    p_m5 = m5.predict_proba(Xte)[:, 1]

    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    n, diffs = len(yte), []
    for _ in range(C.N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if yte[idx].min() == yte[idx].max():
            continue
        diffs.append(average_precision_score(yte[idx], p_hgt[idx])
                     - average_precision_score(yte[idx], p_m5[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out["tests"]["paired_bootstrap_seed0_hgt_minus_m5"] = {
        "point": round(float(average_precision_score(yte, p_hgt)
                             - average_precision_score(yte, p_m5)), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_boot": C.N_BOOTSTRAP}

    suffix = "" if args.variant == "standard" else f"_{args.variant}"
    (C.RESULTS_DIR / f"e12_hgt_vs_baselines{suffix}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
