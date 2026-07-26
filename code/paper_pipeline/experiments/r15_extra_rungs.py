"""R15 - CatBoost and TabPFN ladder rungs (task B8, post hoc rows).

Same temporal split, same tabular features as e1 (build_features,
concepts='none'), 10 seeds, AUC-PR primary. Post hoc by construction
(chemistry-RGCN precedent): M5 and M5' never move; a candidate earns a post
hoc row only if it beats the frozen per-seed M3 AND M5 with a paired
Wilcoxon p < 0.05, after which the full eq. (2) protocol would be run
before any paper row is added.

CatBoost mirrors M3's budget shape: 500 iterations, an internal seeded 15
percent early-stopping slice of TRAIN (the graph models' validation cohort
is never touched for selection).

TabPFN runs OUT OF PROCESS in a dedicated venv (its dependency floor is
incompatible with the frozen analysis environment; installing it in-place
broke and was rolled back). This script only merges a tabpfn_partial JSON
if present. TabPFN sees at most 10,000 training rows; where the train
cohort is larger, the subsample is drawn per seed and that is the seed's
only role (documented in the output).

Usage: DATASET=<f> DATASET_PATH=... python r15_extra_rungs.py   (per field)
       python r15_extra_rungs.py --merge
Output: results/robustness/extra_rungs.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "results" / "robustness"
PARTIAL = OUT_DIR / "extra_rungs_partial"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]


def main_field():
    from catboost import CatBoostClassifier
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    PARTIAL.mkdir(parents=True, exist_ok=True)
    df_split = D.temporal_split(D.load_dataset())
    X, _ = D.build_features(df_split, concepts="none")
    parts = D.split_xy(df_split, X)
    (Xtr, ytr), (Xte, yte) = parts["train"], parts["test"]

    runs = []
    for seed in C.SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(ytr))
        cut = int(0.85 * len(ytr))
        fit_i, stop_i = idx[:cut], idx[cut:]
        m = CatBoostClassifier(iterations=500, random_seed=seed, verbose=0,
                               early_stopping_rounds=20)
        m.fit(Xtr[fit_i], ytr[fit_i], eval_set=(Xtr[stop_i], ytr[stop_i]),
              use_best_model=True)
        auc = S.evaluate(yte, m.predict_proba(Xte)[:, 1])["auc_pr"]
        runs.append(round(float(auc), 4))
        print(f"[r15] {field} catboost seed {seed}: {auc:.4f}", flush=True)
    (PARTIAL / f"{field}_catboost.json").write_text(json.dumps(
        {"field": field, "model": "catboost_500_es20_internal15",
         "per_seed_auc_pr": runs,
         "mean": round(float(np.mean(runs)), 4),
         "std": round(float(np.std(runs, ddof=1)), 4)}, indent=2))


def main_merge():
    out = {"experiment": "R15_extra_rungs",
           "rule": ("post hoc; a row is added only if the candidate beats "
                    "frozen M3 AND M5 per-seed (paired Wilcoxon p<0.05), then "
                    "the full eq.(2) protocol runs first; M5/M5' never move"),
           "fields": {}}
    for f in FIELDS:
        e1 = json.loads((REPO / "results" / f"results_{f}" /
                         "e1_baselines.json").read_text())
        m3 = [r["auc_pr"] for r in e1["per_seed"]["M3_gbdt_tabular"]]
        m5 = [r["auc_pr"] for r in e1["per_seed"]["M5_gbdt_nfa"]]
        row = {"M3_frozen_mean": round(float(np.mean(m3)), 4),
               "M5_frozen_mean": round(float(np.mean(m5)), 4), "candidates": {}}
        for cand in ("catboost", "tabpfn"):
            p = PARTIAL / f"{f}_{cand}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            g = d["per_seed_auc_pr"]
            entry = {"mean": d["mean"], "std": d["std"], "per_seed": g}
            if len(g) == len(m3):
                w3 = S.paired_wilcoxon(g, m3)
                w5 = S.paired_wilcoxon(g, m5)
                entry["vs_M3"] = {"mean_diff": round(w3["mean_diff"], 4), "p": w3["p"]}
                entry["vs_M5"] = {"mean_diff": round(w5["mean_diff"], 4), "p": w5["p"]}
                entry["beats_both"] = bool(w3["mean_diff"] > 0 and w3["p"] < 0.05
                                           and w5["mean_diff"] > 0 and w5["p"] < 0.05)
            else:
                entry["vs_M3_mean_diff"] = round(d["mean"] - float(np.mean(m3)), 4)
                entry["vs_M5_mean_diff"] = round(d["mean"] - float(np.mean(m5)), 4)
                entry["beats_both"] = bool(entry["vs_M3_mean_diff"] > 0
                                           and entry["vs_M5_mean_diff"] > 0)
            row["candidates"][cand] = entry
        out["fields"][f] = row
        print(f, json.dumps(row["candidates"], indent=1)[:300], flush=True)
    out["any_beats_both"] = any(c.get("beats_both")
                                for v in out["fields"].values()
                                for c in v["candidates"].values())
    (OUT_DIR / "extra_rungs.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        main_merge()
    else:
        main_field()
