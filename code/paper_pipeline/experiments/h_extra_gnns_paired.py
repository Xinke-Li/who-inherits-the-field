"""H-paired - paper-style paired significance test for the extra GNNs vs the
tabular-ceiling components M2/M3 on the frozen MAIN split (R3 follow-up).

Reuses e12_hgt_vs_baselines.py's exact machinery (utils.stats.paired_wilcoxon +
utils.stats.bh_correction). For each discipline and each extra architecture
(RGCN, GAT+cohort-time) we test the 10-seed test AUC-PR against M3_gbdt_tabular
and M2_logit_tabular - the two components of the tabular ceiling max(M2, M3)
(econ ceiling = M2, neuro ceiling = M3). The headline is neuro RGCN vs M3.

Per-seed inputs (existing artifacts only; no recompute, no load_dataset, so the
config econ-path is irrelevant):
  model AUC-PR : results_<disc>/results_extra_gnns/<model>_seed<k>.json ["test"]["auc_pr"]
  M2/M3 AUC-PR : results_<disc>/e1_baselines.json  ["per_seed"][key][k]["auc_pr"]
Seeds 0-9 pair naturally (same frozen split, same seed grid), exactly as
e12_hgt_vs_baselines.py pairs HGT vs its baselines.

METHOD, stated plainly (no fudging):
  * M3 (GBDT) is stochastic across seeds -> a genuine PAIRED per-seed Wilcoxon
    signed-rank on the 10 (model - M3) differences.
  * M2 (logit) is DETERMINISTIC (zero seed variance) -> the signed-rank on the 10
    (model - M2) differences reduces to a ONE-SAMPLE signed-rank of the model's
    10 AUC-PRs against the fixed M2 value; reported as such.
  * BH is applied WITHIN each model's {vs M3, vs M2} family, per discipline
    (one model's baseline set = one family, matching the paper).
  * The paper's second test is a seed-0 SCORE-level paired bootstrap. That needs
    the model's per-example seed-0 test scores, which the extra-GNN runs did NOT
    persist (only per-seed AUC-PR). We therefore report, as the interval, a
    SEED-level paired bootstrap CI of the mean AUC-PR difference over the 10
    seeds (2000 resamples) plus a Student-t CI, and flag the score-level CI as
    unavailable-without-rerun. No score-level number is fabricated.

Output: results_<disc>/h_extra_gnns_paired.json (+ printed summary).
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
from utils import stats as S   # reuse the paper's paired_wilcoxon + bh_correction

SEEDS = list(range(10))
N_BOOT = 2000
MODELS = [("rgcn", "RGCN"), ("tgat", "GAT+cohort-time")]
BASES = [("M3_gbdt_tabular", "M3"), ("M2_logit_tabular", "M2")]
CEILING_COMPONENT = {"econ": "M2_logit_tabular", "neuro": "M3_gbdt_tabular"}


def model_perseed(disc, model):
    fs = sorted(glob.glob(os.path.join(
        PP, f"results_{disc}", "results_extra_gnns", f"{model}_seed*.json")))
    assert len(fs) == len(SEEDS), f"{disc}/{model}: {len(fs)} seeds"
    return [json.load(open(f))["test"]["auc_pr"] for f in fs]


def base_perseed(disc, key):
    per = json.load(open(os.path.join(PP, f"results_{disc}", "e1_baselines.json")))["per_seed"]
    return [per[key][i]["auc_pr"] for i in range(len(SEEDS))]


def seedlevel_bootstrap_ci(diffs, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    n = len(d)
    means = [float(np.mean(d[rng.integers(0, n, n)])) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def t_ci(diffs):
    d = np.asarray(diffs, float)
    n = len(d)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    h = float(sps.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
    return m - h, m + h


def analyse(disc):
    ceil_key = CEILING_COMPONENT[disc]
    out = {
        "experiment": "H_extra_gnns_paired",
        "discipline": disc,
        "split": "frozen main temporal split (identical to E1/E2)",
        "n_seeds": len(SEEDS),
        "tabular_ceiling_component": {"key": ceil_key,
                                      "label": "M3" if ceil_key.startswith("M3") else "M2"},
        "method": {
            "paired_test": "utils.stats.paired_wilcoxon (per-seed Wilcoxon signed-rank, "
                           "two-sided), seeds 0-9 paired by index",
            "m2_note": "M2 deterministic (zero seed variance) -> one-sample signed-rank "
                       "of the model's 10 AUC-PRs vs the fixed M2 value",
            "multiplicity": "BH (utils.stats.bh_correction) within each model's {vs M3, vs M2} "
                            "family, per discipline",
            "interval": "seed-level paired bootstrap of mean AUC-PR difference over 10 seeds "
                        "(2000 resamples, 2.5/97.5 pct) + Student-t CI",
            "score_level_bootstrap": "UNAVAILABLE - extra-GNN runs persisted only per-seed "
                                     "AUC-PR, not per-example scores; not fabricated",
        },
        "comparisons": {},
    }
    for model, mlabel in MODELS:
        mpr = model_perseed(disc, model)
        fam = []
        for key, blabel in BASES:
            bpr = base_perseed(disc, key)
            w = S.paired_wilcoxon(mpr, bpr)
            diffs = np.asarray(mpr, float) - np.asarray(bpr, float)
            bl, bh = seedlevel_bootstrap_ci(diffs)
            tl, th = t_ci(diffs)
            deterministic_base = float(np.var(bpr)) == 0.0
            fam.append({
                "model": model, "model_label": mlabel,
                "baseline": key, "baseline_label": blabel,
                "is_ceiling_component": key == ceil_key,
                "model_mean": round(float(np.mean(mpr)), 4),
                "baseline_mean": round(float(np.mean(bpr)), 4),
                "mean_diff_model_minus_base": round(w["mean_diff"], 4),
                "n_pos_diffs": int((diffs > 0).sum()),
                "n_neg_diffs": int((diffs < 0).sum()),
                "wilcoxon_p_raw": round(float(w["p"]), 6),
                "test_kind": ("one-sample signed-rank vs deterministic M2"
                              if deterministic_base else
                              "paired per-seed signed-rank (both stochastic)"),
                "ci95_seedlevel_bootstrap": [round(bl, 4), round(bh, 4)],
                "ci95_t": [round(tl, 4), round(th, 4)],
                "ci_crosses_zero": bool(bl <= 0 <= bh),
            })
        # BH within this model's family (vs M3, vs M2)
        p_adj, reject = S.bh_correction([c["wilcoxon_p_raw"] for c in fam])
        for c, pa, rj in zip(fam, p_adj, reject):
            c["p_adj_bh_within_model_family"] = round(float(pa), 6)
            c["significant_bh_0.05"] = bool(rj)
        out["comparisons"][model] = fam

    # headline verdict: the ceiling comparison for each model
    hv = {}
    for model, _ in MODELS:
        c = next(x for x in out["comparisons"][model] if x["is_ceiling_component"])
        sig = c["significant_bh_0.05"] and not c["ci_crosses_zero"]
        hv[model] = {
            "vs_ceiling_component": c["baseline_label"],
            "mean_diff": c["mean_diff_model_minus_base"],
            "p_adj": c["p_adj_bh_within_model_family"],
            "ci95_seedlevel_bootstrap": c["ci95_seedlevel_bootstrap"],
            "exceeds_ceiling_point_estimate": c["mean_diff_model_minus_base"] > 0,
            "significant_exceedance": bool(c["mean_diff_model_minus_base"] > 0 and sig),
        }
    out["headline_vs_ceiling_component"] = hv
    return out


def main():
    for disc in ("econ", "neuro"):
        out = analyse(disc)
        outp = os.path.join(PP, f"results_{disc}", "h_extra_gnns_paired.json")
        json.dump(out, open(outp, "w"), indent=2)
        ck = out["tabular_ceiling_component"]["label"]
        print(f"\n===== {disc.upper()}  (ceiling component = {ck}) =====")
        for model, mlabel in MODELS:
            for c in out["comparisons"][model]:
                star = " *CEILING*" if c["is_ceiling_component"] else ""
                sig = "SIG" if c["significant_bh_0.05"] else "ns"
                print(f"  {mlabel:<16s} vs {c['baseline_label']:<2s}{star:<9s} "
                      f"Δ {c['mean_diff_model_minus_base']:+.4f}  "
                      f"p={c['wilcoxon_p_raw']:.4g} p_adj={c['p_adj_bh_within_model_family']:.4g} "
                      f"[{sig}]  boot95 [{c['ci95_seedlevel_bootstrap'][0]:+.4f},"
                      f"{c['ci95_seedlevel_bootstrap'][1]:+.4f}] "
                      f"(pos {c['n_pos_diffs']}/{c['n_pos_diffs']+c['n_neg_diffs']}, "
                      f"{c['test_kind'].split('(')[0].strip()})")
        hv = out["headline_vs_ceiling_component"]
        for model, mlabel in MODELS:
            v = hv[model]
            verdict = ("SIGNIFICANTLY EXCEEDS" if v["significant_exceedance"]
                       else "point estimate above but NOT significant" if v["exceeds_ceiling_point_estimate"]
                       else "at/below ceiling")
            print(f"  -> {mlabel} vs ceiling({v['vs_ceiling_component']}): {verdict} "
                  f"(Δ {v['mean_diff']:+.4f}, p_adj {v['p_adj']:.4g}, "
                  f"boot95 {v['ci95_seedlevel_bootstrap']})")
        print(f"  wrote {os.path.relpath(outp, PP)}")


if __name__ == "__main__":
    main()
