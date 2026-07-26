"""Regenerate ALL publication figures (matplotlib/seaborn, SEC-notebook
aesthetic) from results_econ/*.json. This is the authoritative figure pass;
utils/mplfigs.py holds the style. PNG (300 dpi) + PDF (vector, for LaTeX)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import mplfigs as M
from utils import stats as S
from utils import data as D

R = C.RESULTS_DIR
load = lambda n: json.load(open(R / n))

d = load("e1_baselines.json")
M.fig_baselines(d["summary"])

d = load("e10_advisor_placebo.json")
M.fig_advisor_placebo(d["summary"])

g = [json.loads(l)["metrics"] for l in open(R / "e9a_global_perseed.jsonl")]
c = [json.loads(l)["metrics"] for l in open(R / "e9a_perseed.jsonl")]
M.fig_placebo(g, c)

d = load("e9_robustness.json")
M.fig_robustness({k: {"summary": v} for k, v in d.items() if k != "experiment"})

d = load("e4_label_robustness.json")
M.fig_theta_stability(d["theta_sweep"])
thetas = sorted(d["theta_sweep"], key=float)
M.fig_theta_single([float(t) for t in thetas],
                   [d["theta_sweep"][t]["auc_pr"]["mean"] for t in thetas])

d = load("e6_innovation_premium.json")
M.fig_coefficient_ladder(d["primary_outcome"]["late_cite_pct_mean"])

d = load("e8_heterogeneity.json")
M.fig_cohort_trends(d["cohort_trends"])

d = load("e3_ablation.json")
M.fig_forest(d["comparisons"])

d = load("e0_descriptive.json")
M.fig_top50(d["top50"])

df = D.load_dataset()
q = pd.qcut(df.early_overlap, 5, duplicates="drop")
gq = df.groupby(q, observed=True).y.agg(["mean", "count"])
M.fig_quintiles([f"Q{i+1}" for i in range(len(gq))], list(gq["mean"]), list(gq["count"]))

d0 = json.loads((R / "results_hgt" / "hgt_none_seed0.json").read_text())
M.fig_hgt_training(d0["history"])
y, p = d0["test_labels"], d0["test_scores"]
M.fig_hgt_roc(y, p)
M.fig_hgt_pr(y, p)
thr = S.best_f1_threshold(np.array(d0["val_labels"]), np.array(d0["val_scores"]))
M.fig_hgt_confusion(y, p, thr)

print("all publication figures regenerated (png + pdf)")
