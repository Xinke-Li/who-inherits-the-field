"""E2-rolling - HGT per rolling-origin cohort (GPU leg of E12-rolling; R2).

Thin wrapper over e2_hgt.py: SAME model, hyperparameters, early stopping,
and seed protocol; the ONLY change is the train/val/test partition, which is
read from the manifest emitted by e12_rolling_origin.py so the GPU runs score
exactly the pre-registered cohorts.

Usage (one run = one origin x one seed):
  python e2_hgt_rolling.py --origin 2003-2004 --seed 0 \
      [--manifest ../results_econ/e12_rolling_origin_manifest_econ.json] \
      [--out ../results_econ/results_hgt_rolling]
NEURO_DATASET env switches discipline exactly as elsewhere.

Output: <out>/hgt_<origin>_seed<k>.json  (same schema as e2_hgt.py)
Collect with e12_collect_hgt_rolling.py after all origins x seeds finish.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from utils import data as D
import e2_hgt as E2

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", required=True, help="e.g. 2003-2004")
    ap.add_argument("--seed", type=int, default=0)
    _disc = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
    ap.add_argument("--manifest", default=str(C.RESULTS_DIR / f"e12_rolling_origin_manifest_{_disc}.json"))
    ap.add_argument("--out", default=str(C.RESULTS_DIR / "results_hgt_rolling"))
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    assert args.origin in man, f"origin {args.origin} not in manifest ({list(man)})"
    fold = man[args.origin]

    E2.set_seed(args.seed)
    df = D.load_dataset()
    pid2split = {p: "train" for p in fold["train_pids"]}
    pid2split.update({p: "val" for p in fold["val_pids"]})
    pid2split.update({p: "test" for p in fold["test_pids"]})
    df = df[df.student_pid.isin(pid2split)].reset_index(drop=True)
    df["split"] = df.student_pid.map(pid2split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[e2-rolling] WARNING: no CUDA - this is the GPU leg; expect hours on CPU")
    data = E2.build_graph(df, "none")
    res = E2.train_eval(data, args.seed, device)
    res.update({"origin": args.origin, "ablate": "none",
                "n": len(df), "protocol": "e12_rolling_origin manifest split"})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"hgt_{args.origin}_seed{args.seed}.json")
    Path(path).write_text(json.dumps(res, indent=2))
    print(f"[e2-rolling] {args.origin} seed {args.seed} -> {path} "
          f"(test auc_pr {res.get('auc_pr')})")


if __name__ == "__main__":
    main()
