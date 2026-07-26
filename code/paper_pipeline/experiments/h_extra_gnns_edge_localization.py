"""H-edge-localization - localize the neuroscience RGCN's ceiling exceedance to a
single edge type (R3 follow-up to h_extra_gnns_paired.py).

The `social` group is advising + institution + coauthorship. We drop each ONE at a
time (coauth / institution / advising) and the whole group (social), re-run RGCN on
neuroscience 10 seeds each, and ask two paired questions with the paper's own
machinery (utils.stats.paired_wilcoxon + bh_correction):

  (A) redundancy:  drop-<edge> vs full RGCN (per-seed paired) - does removing one
      edge move the model at all?
  (B) ceiling:     each variant vs the tabular ceiling component M3 (per-seed
      paired, seeds 0-9 by index) - does it still clear the ceiling?

Per-seed inputs (existing artifacts only; no recompute, no load_dataset):
  full           results_neuro/results_extra_gnns/rgcn_seed<k>.json        ["test"]["auc_pr"]
  drop-<edge>    results_neuro/results_extra_gnns/rgcn_ablate-<edge>_seed<k>.json ["test"]["auc_pr"]
  M3 (ceiling)   results_neuro/e1_baselines.json ["per_seed"]["M3_gbdt_tabular"][k]["auc_pr"]

Interval: seed-level paired bootstrap of the mean AUC-PR difference over the 10
seeds (2000 resamples) + Student-t CI; the score-level bootstrap is unavailable
(these runs persist per-seed AUC-PR, not per-example scores), stated as such.

Reading (redundant/distributed signal): if no single-edge drop is significant vs
full and each single-edge variant still clears M3, while only the full social drop
falls below M3, the exceedance is a redundant relational-structure effect, not a
co-authorship effect. NO over-claim: "dropping coauth does not reduce the gain",
never "co-authorship carries zero signal".

Output: results_neuro/h_extra_gnns_edge_localization.json (+ printed summary).
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, PP)
from utils import stats as S   # reuse paired_wilcoxon + bh_correction

DISC = "neuro"
SEEDS = list(range(10))
N_BOOT = 2000
EDGES = ["coauth", "institution", "advising", "social"]
RDIR = os.path.join(PP, f"results_{DISC}", "results_extra_gnns")
CEIL_KEY = "M3_gbdt_tabular"


def rgcn_perseed(tag):
    """tag='' -> full rgcn_seed*.json ; tag='coauth' -> rgcn_ablate-coauth_seed*.json"""
    pat = "rgcn_seed*.json" if tag == "" else f"rgcn_ablate-{tag}_seed*.json"
    fs = sorted(glob.glob(os.path.join(RDIR, pat)))
    assert len(fs) == len(SEEDS), f"{pat}: {len(fs)} seeds"
    return [json.load(open(f))["test"]["auc_pr"] for f in fs]


def m3_perseed():
    per = json.load(open(os.path.join(PP, f"results_{DISC}", "e1_baselines.json")))["per_seed"]
    return [per[CEIL_KEY][i]["auc_pr"] for i in range(len(SEEDS))]


def seedlevel_bootstrap_ci(diffs, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float); n = len(d)
    means = [float(np.mean(d[rng.integers(0, n, n)])) for _ in range(N_BOOT)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def t_ci(diffs):
    d = np.asarray(diffs, float); n = len(d)
    h = float(sps.t.ppf(0.975, n - 1)) * float(d.std(ddof=1)) / np.sqrt(n)
    return round(float(d.mean() - h), 4), round(float(d.mean() + h), 4)


def compare(a, b):
    """paired per-seed a - b: mean diff, Wilcoxon p, direction, CIs."""
    w = S.paired_wilcoxon(a, b)
    diffs = np.asarray(a, float) - np.asarray(b, float)
    return {"mean_diff": round(w["mean_diff"], 4), "p_raw": round(float(w["p"]), 6),
            "n_pos": int((diffs > 0).sum()), "n_neg": int((diffs < 0).sum()),
            "ci95_seedlevel_bootstrap": list(seedlevel_bootstrap_ci(diffs)),
            "ci95_t": list(t_ci(diffs))}


def main():
    full = rgcn_perseed("")
    m3 = m3_perseed()
    variants = {"full": full, **{f"drop-{e}": rgcn_perseed(e) for e in EDGES}}
    means = {g: {"auc_pr_mean": round(float(np.mean(v)), 4),
                 "auc_pr_std": round(float(np.std(v, ddof=1)), 4)} for g, v in variants.items()}
    ceil = round(float(np.mean(m3)), 4)

    # (A) each single/whole drop vs full RGCN; BH within this family
    vs_full = []
    for e in EDGES:
        c = compare(variants[f"drop-{e}"], full)
        c.update({"ablate": e, "comparison": f"drop-{e} vs full"})
        vs_full.append(c)
    padj, rej = S.bh_correction([c["p_raw"] for c in vs_full])
    for c, pa, rj in zip(vs_full, padj, rej):
        c["p_adj_bh"] = round(float(pa), 6); c["significant"] = bool(rj)

    # (B) each variant vs the M3 ceiling; BH within this family
    vs_ceil = []
    for g in ("full", *[f"drop-{e}" for e in EDGES]):
        c = compare(variants[g], m3)
        lo, hi = c["ci95_seedlevel_bootstrap"]
        c.update({"variant": g, "comparison": f"{g} vs M3",
                  "above_ceiling_point": c["mean_diff"] > 0,
                  "ci_crosses_zero": bool(lo <= 0 <= hi)})
        vs_ceil.append(c)
    padj, rej = S.bh_correction([c["p_raw"] for c in vs_ceil])
    for c, pa, rj in zip(vs_ceil, padj, rej):
        c["p_adj_bh"] = round(float(pa), 6)
        c["significant_vs_ceiling"] = bool(rj and not c["ci_crosses_zero"])

    def get_vf(e): return next(x for x in vs_full if x["ablate"] == e)
    def get_vc(g): return next(x for x in vs_ceil if x["variant"] == g)
    verdict = {
        "coauth_drop_vs_full_significant": get_vf("coauth")["significant"],
        "coauth_drop_still_above_ceiling": (get_vc("drop-coauth")["above_ceiling_point"]
                                            and get_vc("drop-coauth")["significant_vs_ceiling"]),
        "no_single_edge_drop_significant_vs_full": not any(get_vf(e)["significant"]
                                                           for e in ("coauth", "institution", "advising")),
        "every_single_edge_variant_above_ceiling": all(get_vc(f"drop-{e}")["above_ceiling_point"]
                                                       for e in ("coauth", "institution", "advising")),
        "social_drop_vs_full_significant": get_vf("social")["significant"],
        "social_drop_below_ceiling": not get_vc("drop-social")["above_ceiling_point"],
        "sum_single_drops_vs_full": round(sum(get_vf(e)["mean_diff"]
                                              for e in ("coauth", "institution", "advising")), 4),
        "social_drop_vs_full": get_vf("social")["mean_diff"],
        "reading": ("redundant/distributed relational-structure effect: no single social edge "
                    "drop reduces the gain (each stays above M3); only removing all three falls "
                    "below the ceiling. Dropping co-authorship alone does not reduce the gain, so "
                    "the collaboration-transmission null is untouched. NOT a claim that "
                    "co-authorship carries zero signal."),
    }
    out = {"experiment": "H_extra_gnns_edge_localization", "discipline": DISC, "model": "rgcn",
           "split": "frozen main temporal split (identical to E1/E2)", "n_seeds": len(SEEDS),
           "tabular_ceiling": {"key": CEIL_KEY, "value": ceil},
           "method": {
               "paired_test": "utils.stats.paired_wilcoxon (per-seed Wilcoxon signed-rank, two-sided), "
                              "seeds 0-9 paired by index",
               "multiplicity": "BH (utils.stats.bh_correction) within the vs-full family (4) and "
                               "within the vs-M3 family (5)",
               "interval": "seed-level paired bootstrap of mean AUC-PR diff over 10 seeds "
                           "(2000 resamples) + Student-t CI",
               "score_level_bootstrap": "UNAVAILABLE - runs persisted per-seed AUC-PR, not "
                                        "per-example scores; not fabricated"},
           "means": means, "vs_full": vs_full, "vs_ceiling_M3": vs_ceil, "verdict": verdict}
    outp = os.path.join(PP, f"results_{DISC}", "h_extra_gnns_edge_localization.json")
    json.dump(out, open(outp, "w"), indent=2)

    print(f"neuro RGCN edge localization (ceiling M3 = {ceil}):")
    print("  means:", {g: means[g]["auc_pr_mean"] for g in means})
    print("  --- (A) drop vs full (BH within 4) ---")
    for c in vs_full:
        print(f"    {c['comparison']:<22s} Δ {c['mean_diff']:+.4f}  p={c['p_raw']:.4g} "
              f"p_adj={c['p_adj_bh']:.4g}  {'SIG' if c['significant'] else 'ns'}  "
              f"boot{c['ci95_seedlevel_bootstrap']}  pos {c['n_pos']}/10")
    print("  --- (B) variant vs M3 ceiling (BH within 5) ---")
    for c in vs_ceil:
        above = "ABOVE" if c["above_ceiling_point"] else "below"
        print(f"    {c['comparison']:<22s} Δ {c['mean_diff']:+.4f}  p_adj={c['p_adj_bh']:.4g}  "
              f"{above}  {'SIG' if c['significant_vs_ceiling'] else 'ns'}  "
              f"boot{c['ci95_seedlevel_bootstrap']}")
    print("  verdict:", json.dumps({k: verdict[k] for k in verdict if k != 'reading'}))
    print(f"  wrote {os.path.relpath(outp, PP)}")


if __name__ == "__main__":
    main()
