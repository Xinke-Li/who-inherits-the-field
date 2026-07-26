"""E3 - Aggregate HGT runs: paired tests + BH correction + forest plot.

Run AFTER the Colab jobs finish and results_hgt/*.json are synced locally.
Pre-registered decision rule (restated from the analysis plan):
  "removing social edges improves performance" is kept ONLY if the paired
  Wilcoxon (10 seeds) survives BH at alpha=.05 AND the bootstrap CI excludes 0.
  Otherwise the claim is restated as "social edges carry no additional
  predictive information".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import stats as S

HGT_DIR = C.RESULTS_DIR / "results_hgt"   # sync Colab outputs here


def load_runs():
    runs = {}
    for p in sorted(HGT_DIR.glob("hgt_*_seed*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["ablate"], {})[r["seed"]] = r
    return runs


def main():
    runs = load_runs()
    if "none" not in runs:
        sys.exit(f"no full-model runs found in {HGT_DIR} - run e2_hgt.py first")
    seeds = sorted(runs["none"])
    print(f"[e3] found ablations={sorted(runs)} | seeds={seeds}")

    full = [runs["none"][s]["auc_pr"] for s in seeds]
    rows, pvals = [], []
    for ab in sorted(k for k in runs if k != "none"):
        missing = [s for s in seeds if s not in runs[ab]]
        if missing:
            print(f"[e3] WARNING: {ab} missing seeds {missing} - skipped")
            continue
        abl = [runs[ab][s]["auc_pr"] for s in seeds]
        t = S.paired_wilcoxon(abl, full)
        d = np.array(t["diffs"])
        rng = np.random.default_rng(0)
        boots = [np.mean(rng.choice(d, len(d))) for _ in range(C.N_BOOTSTRAP)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"ablation": ab, "mean_diff": t["mean_diff"],
                     "ci95": [float(lo), float(hi)], "p_raw": t["p"],
                     "cohens_d": S.cohens_d_paired(abl, full),
                     "full_mean": float(np.mean(full)), "abl_mean": float(np.mean(abl))})
        pvals.append(t["p"])

    adj, reject = S.bh_correction(pvals)
    for r, p_adj, rej in zip(rows, adj, reject):
        r["p_adj"] = p_adj
        r["significant"] = bool(rej)
        r["verdict"] = ("claim supported" if rej and
                        (r["ci95"][0] > 0 or r["ci95"][1] < 0)
                        else "NOT significant - use weakened wording")

    out = {"experiment": "E3_ablation", "seeds": seeds,
           "full_model_auc_pr": {"mean": float(np.mean(full)),
                                 "std": float(np.std(full, ddof=1))},
           "comparisons": rows}
    (C.RESULTS_DIR / "e3_ablation.json").write_text(json.dumps(out, indent=2))

    for r in rows:
        print(f"  {r['ablation']:<18} Δ={r['mean_diff']:+.4f} CI=[{r['ci95'][0]:+.4f},"
              f"{r['ci95'][1]:+.4f}] p_adj={r['p_adj']:.4f} -> {r['verdict']}")

    try:
        from utils import plotting as P
        P.plot_ablation_forest([r["ablation"] for r in rows],
                               [r["mean_diff"] for r in rows],
                               [r["ci95"][0] for r in rows],
                               [r["ci95"][1] for r in rows],
                               [r["p_adj"] for r in rows],
                               fname="F7e_ablation_forest")
    except Exception as e:
        print(f"[e3] plotting skipped: {e}")


if __name__ == "__main__":
    main()
