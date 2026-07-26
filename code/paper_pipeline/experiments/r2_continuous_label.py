"""R2 - Continuous-label check: regress late_overlap directly (P0-2).

POST-HOC ROBUSTNESS CHECK, layered on top of the frozen artifact; NOT part of
the pre-registered protocol. Removes the threshold entirely: the target is the
continuous late_overlap, on the same temporal split and the same pre-window
features as the classification ladder. Regressor analogues of the ladder:

  M1r  OLS on the single early_overlap scalar         (M1 analogue)
  M2r  ridge on all tabular features                  (M2 analogue)
  M3r  HistGradientBoostingRegressor, same features   (M3 analogue)
  M5r  HistGradientBoostingRegressor + the temporally-guarded NFA features
       (M5 analogue; build_nfa_features imported verbatim - its
       nfa_adv_track_retention aggregate keeps the frozen theta=0.2 sibling
       label, which is a legitimate pre-window feature transformation and is
       noted here rather than redefined)

10 seeds (M1r/M2r deterministic given the split; seed drives the boosting
models), Spearman rho (primary) and R^2 on the test cohort. The question the
appendix table answers: does the M1 <= M2 <= M3 <= M5 ordering and the
"M1 recovers most of the signal" claim survive with no threshold at all?

Usage (one discipline per process):
  DATASET=<field> DATASET_PATH=data/clean_dataset_<field>.parquet \
      python code/paper_pipeline/experiments/r2_continuous_label.py
  python code/paper_pipeline/experiments/r2_continuous_label.py --merge

Outputs: results/robustness/continuous_partial/<field>.json (per process),
         results/robustness/continuous_label_summary.json (--merge).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D

from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "results" / "robustness"
PARTIAL_DIR = OUT_DIR / "continuous_partial"
FIELDS = ["chemistry", "econ", "math", "neuro", "physics"]


def eval_reg(y_true, pred):
    rho = spearmanr(y_true, pred).statistic
    return {"spearman_rho": round(float(rho), 4),
            "r2": round(float(r2_score(y_true, pred)), 4)}


def main_field():
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    df = D.load_dataset()
    df_split = D.temporal_split(df)
    target = df_split[C.CONTINUOUS_LABEL].astype(float)

    Xt, cols = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)  # frozen y feeds the retention aggregate
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    i_ov = cols.index("early_overlap")

    masks = {k: (df_split.split == k).values for k in ("train", "val", "test")}
    yc = {k: target.values[m] for k, m in masks.items()}
    yte = yc["test"]

    per_model = {}
    # deterministic models: one fit
    m1 = LinearRegression().fit(Xt[masks["train"]][:, [i_ov]], yc["train"])
    per_model["M1r_ols_overlap"] = [eval_reg(yte, m1.predict(Xt[masks["test"]][:, [i_ov]]))]
    m2 = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xt[masks["train"]], yc["train"])
    per_model["M2r_ridge_tabular"] = [eval_reg(yte, m2.predict(Xt[masks["test"]]))]
    # boosting models: 10 seeds, e1's boosting configuration transposed to regression
    for name, X in (("M3r_gbr_tabular", Xt), ("M5r_gbr_nfa", X5)):
        runs = []
        for seed in C.SEEDS:
            m = HistGradientBoostingRegressor(random_state=seed, max_iter=500,
                                              early_stopping=True, validation_fraction=0.15)
            m.fit(X[masks["train"]], yc["train"])
            runs.append(eval_reg(yte, m.predict(X[masks["test"]])))
        per_model[name] = runs

    def summ(runs, key):
        v = np.array([r[key] for r in runs])
        return {"mean": round(float(v.mean()), 4),
                "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)}

    summary = {m: {k: summ(v, k) for k in ("spearman_rho", "r2")}
               for m, v in per_model.items()}
    order = ["M1r_ols_overlap", "M2r_ridge_tabular", "M3r_gbr_tabular", "M5r_gbr_nfa"]
    rhos = [summary[m]["spearman_rho"]["mean"] for m in order]
    out = {"experiment": "R2_continuous_label", "field": field,
           "target": C.CONTINUOUS_LABEL, "seeds": C.SEEDS,
           "summary": summary, "per_seed": per_model,
           "ladder_ordering_rho": {"values": rhos,
                                   "monotone_nondecreasing": bool(all(
                                       rhos[i] <= rhos[i + 1] + 1e-12 for i in range(3)))},
           "m1_share_of_best_rho": round(rhos[0] / max(rhos), 4) if max(rhos) > 0 else None}
    p = PARTIAL_DIR / f"{field}.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("field", "summary", "ladder_ordering_rho",
                                          "m1_share_of_best_rho")}, indent=2))


def main_merge():
    merged = {"experiment": "R2_continuous_label",
              "note": "post-hoc robustness; no threshold; same split and features",
              "fields": {}}
    for f in FIELDS:
        merged["fields"][f] = json.loads((PARTIAL_DIR / f"{f}.json").read_text())
    (OUT_DIR / "continuous_label_summary.json").write_text(json.dumps(merged, indent=2))
    print("[r2] merged ->", OUT_DIR / "continuous_label_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        main_merge()
    else:
        main_field()
