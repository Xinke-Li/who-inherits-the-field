"""H-summary - 10-seed mean/std of the extra GNNs vs the tabular ceiling (R3).

Reads ONLY result JSONs (never the frozen table; no load_dataset, so the config
econ-path is irrelevant here). For each discipline and each extra architecture:

  rgcn  -> label "RGCN"              (HeteroConv/SAGEConv, relation-specific)
  tgat  -> label "GAT+cohort-time"  (GATConv + a per-student cohort-time channel;
           a pragmatic temporal-attention variant, NOT a full temporal GNN /
           TGN - reported honestly under that name)

Each seed file is results_<disc>/results_extra_gnns/<model>_seed<k>.json with the
E2-schema test block: json["test"]["auc_pr"] / ["auc_roc"] (nested, unlike the
Leg-1 rolling HGT which is top-level). We aggregate the 10 seeds and compare the
mean test AUC-PR to the frozen main-split tabular ceiling max(M2, M3), read from
results_<disc>/e1_baselines.json (identical to the F2 ladder's ceiling). Std is
the sample sd (ddof=1), matching e1_baselines' stored convention so the ladder
error bars are consistent.

Decision rule (pre-registered in h_extra_gnns.py): these architectures update the
"no measurable graph gain" reading only if one EXCEEDS the ceiling on a 10-seed
mean. exceeds_ceiling is surfaced per model so any breach is impossible to miss.

Output: results_<disc>/h_extra_gnns_summary.json (+ a printed table).
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PP = os.path.normpath(os.path.join(HERE, ".."))            # paper_pipeline/
DISCS = ("econ", "neuro")
MODELS = [("rgcn", "RGCN"), ("tgat", "GAT+cohort-time")]   # honest tgat label
NSEEDS = 10
DDOF = 1                                                    # match e1_baselines


def ceiling_from_e1(disc):
    s = json.load(open(os.path.join(PP, f"results_{disc}", "e1_baselines.json")))["summary"]
    m2 = s["M2_logit_tabular"]["auc_pr"]["mean"]
    m3 = s["M3_gbdt_tabular"]["auc_pr"]["mean"]
    return max(m2, m3), m2, m3


def model_stats(disc, model):
    fs = sorted(glob.glob(os.path.join(
        PP, f"results_{disc}", "results_extra_gnns", f"{model}_seed*.json")))
    assert len(fs) == NSEEDS, f"{disc}/{model}: expected {NSEEDS} seeds, got {len(fs)}"
    pr, roc = [], []
    for f in fs:
        d = json.load(open(f))
        pr.append(d["test"]["auc_pr"])      # nested E2 test schema
        roc.append(d["test"]["auc_roc"])
    return (float(np.mean(pr)), float(np.std(pr, ddof=DDOF)),
            float(np.mean(roc)), float(np.std(roc, ddof=DDOF)))


def main():
    for disc in DISCS:
        ceil_v, m2, m3 = ceiling_from_e1(disc)
        out = {
            "experiment": "H_extra_gnns_summary",
            "discipline": disc,
            "split": "temporal (frozen main split; identical to E2/E1)",
            "n_seeds": NSEEDS,
            "std_convention": "sample sd, np.std(ddof=1) (matches e1_baselines)",
            "metric": "test AUC-PR (auc_roc also recorded)",
            "tabular_ceiling": {
                "value": round(ceil_v, 4),
                "source": "max(M2_logit_tabular, M3_gbdt_tabular) auc_pr mean, "
                          "results_%s/e1_baselines.json" % disc,
                "M2_logit_tabular": round(m2, 4),
                "M3_gbdt_tabular": round(m3, 4),
            },
            "models": {},
        }
        print(f"\n[{disc}] tabular ceiling max(M2,M3) = {ceil_v:.4f} "
              f"(M2 {m2:.4f}, M3 {m3:.4f})")
        for key, label in MODELS:
            pm, ps, rm, rs = model_stats(disc, key)
            delta = pm - ceil_v
            exceeds = pm > ceil_v
            out["models"][key] = {
                "label": label,
                "family": "graph-aware",
                "auc_pr_mean": round(pm, 4), "auc_pr_std": round(ps, 4),
                "auc_roc_mean": round(rm, 4), "auc_roc_std": round(rs, 4),
                "delta_vs_ceiling": round(delta, 4),
                "exceeds_ceiling": exceeds,
                "n_seeds": NSEEDS,
            }
            flag = "  <-- EXCEEDS CEILING" if exceeds else ""
            print(f"   {label:<16s} AUC-PR {pm:.4f} +- {ps:.4f}  "
                  f"(Δ vs ceiling {delta:+.4f}){flag}")
        any_exceed = any(v["exceeds_ceiling"] for v in out["models"].values())
        out["any_model_exceeds_ceiling"] = any_exceed
        outp = os.path.join(PP, f"results_{disc}", "h_extra_gnns_summary.json")
        with open(outp, "w") as f:
            json.dump(out, f, indent=2)
        print(f"   wrote {os.path.relpath(outp, PP)} "
              f"(any model exceeds ceiling: {any_exceed})")


if __name__ == "__main__":
    main()
