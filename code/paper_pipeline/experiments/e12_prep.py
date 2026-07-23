"""Chunked, resumable per-seed baseline AUCs for E12 (workaround for a
corrupted e1_baselines.json sync). One JSON line per seed in
results_econ/e12_prep.jsonl; finished seeds are skipped."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT = C.RESULTS_DIR / "e12_prep.jsonl"
done = set()
if OUT.exists():
    done = {json.loads(l)["seed"] for l in OUT.read_text().splitlines() if l.strip()}
todo = [s for s in C.SEEDS if s not in done][: int(sys.argv[1]) if len(sys.argv) > 1 else 2]
if not todo:
    print("all seeds done"); sys.exit(0)

df = D.load_dataset()
df_split = D.temporal_split(df)
Xt, _ = D.build_features(df_split, concepts="none")
nfa = D.build_nfa_features(df_split)
X5 = np.hstack([Xt, nfa.values.astype(float)])
tab = D.split_xy(df_split, Xt); p5 = D.split_xy(df_split, X5)
(Xtr, ytr), (Xva, yva), (Xte, yte) = tab["train"], tab["val"], tab["test"]
(Xtr5, _), _, (Xte5, _) = p5["train"], p5["val"], p5["test"]

for seed in todo:
    m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    m2.fit(Xtr, ytr)
    m3 = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                        early_stopping=True, validation_fraction=0.15)
    m3.fit(Xtr, ytr)
    m5 = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                        early_stopping=True, validation_fraction=0.15)
    m5.fit(Xtr5, ytr)
    row = {"seed": seed,
           "M2_logit_tabular": S.evaluate(yte, m2.predict_proba(Xte)[:, 1])["auc_pr"],
           "M3_gbdt_tabular": S.evaluate(yte, m3.predict_proba(Xte)[:, 1])["auc_pr"],
           "M5_gbdt_nfa": S.evaluate(yte, m5.predict_proba(Xte5)[:, 1])["auc_pr"]}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("seed", seed, {k: round(v, 4) for k, v in row.items() if k != "seed"})
