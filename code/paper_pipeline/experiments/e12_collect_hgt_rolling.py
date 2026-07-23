"""E12-collect - merge per-origin HGT runs into e12_rolling_origin.json (R2).

Run AFTER the GPU batch (e2_hgt_rolling.py over all origins x seeds; see
GPU_Batch_Colab.ipynb / PENDING_TODO.md). For each origin, computes the HGT
10-seed mean/std test AUC-PR and the deltas against the CPU ladder already in
e12_rolling_origin.json, then rewrites that file in place with an
"hgt" block per origin and an updated verdict that covers the HGT leg.
Idempotent; refuses to write if any origin is missing seeds (no partial claims).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C

RES = C.RESULTS_DIR / "e12_rolling_origin.json"
RUNS = C.RESULTS_DIR / "results_hgt_rolling"


def main():
    out = json.loads(RES.read_text())
    seeds = out["seeds"]
    complete = True
    for rec in out["origins"]:
        origin = rec["origin"]
        aucs = []
        for s in seeds:
            p = RUNS / f"hgt_{origin}_seed{s}.json"
            if not p.exists():
                print(f"[collect] MISSING {p.name}"); complete = False; continue
            aucs.append(json.loads(p.read_text())["auc_pr"])  # E2 returns test metrics at top level
        if len(aucs) == len(seeds):
            m, sd = float(np.mean(aucs)), float(np.std(aucs))
            rec["hgt"] = {"auc_pr_mean": round(m, 4), "auc_pr_std": round(sd, 4),
                          "delta_hgt_minus_ceiling": round(m - rec["tabular_ceiling"], 4)}
            print(f"[collect] {origin}: HGT {m:.4f}±{sd:.4f} "
                  f"(Δ vs ceiling {m - rec['tabular_ceiling']:+.4f})")
    if not complete:
        sys.exit("[collect] incomplete - not updating the verdict (no partial claims)")
    deltas = [r["hgt"]["delta_hgt_minus_ceiling"] for r in out["origins"]]
    out["verdict"]["hgt_per_origin"] = {
        "deltas_hgt_minus_ceiling": deltas,
        "sign_consistent_positive": all(d > 0 for d in deltas),
    }
    RES.write_text(json.dumps(out, indent=2))
    print(f"[collect] verdict updated: HGT deltas {deltas}")


if __name__ == "__main__":
    main()
