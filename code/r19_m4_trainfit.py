#!/usr/bin/env python3
"""T2.3 (audit finding W5): rerun M4 with the TF-IDF fitted on the training split.

The frozen path fits the vectorizer on train+val+test. The vocabulary genuinely
comes from pre-window text, but the IDF weights and the max_features cut are
estimated from validation and test rows, which no tabular rung sees. This script
recomputes M4 both ways on the same split, same seeds, same classifier, so the
size of the effect is measured rather than argued.

Nothing frozen is touched. Output: results/revision/T2_3_m4_trainfit/.

  python code/r19_m4_trainfit.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_3_m4_trainfit"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
SEEDS = list(range(10))


def run_one(field):
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from utils import data as D
    from utils import stats as S

    dsp = D.temporal_split(D.load_dataset())
    out = {"field": field, "seeds": SEEDS, "arms": {}}

    for arm, mode in (("trainfit", "tfidf"), ("legacy", "tfidf_legacy")):
        X, names = D.build_features(dsp, concepts=mode)
        sp = D.split_xy(dsp, X)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = sp["train"], sp["val"], sp["test"]
        per_seed = []
        for seed in SEEDS:
            m = make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(max_iter=1000, C=0.5, random_state=seed,
                                   solver="liblinear"))
            m.fit(Xtr, ytr)
            thr = S.best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
            per_seed.append(S.evaluate(yte, m.predict_proba(Xte)[:, 1], thr))
        agg = {k: {"mean": float(sum(d[k] for d in per_seed) / len(per_seed))}
               for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))}
        out["arms"][arm] = {"n_features": int(X.shape[1]),
                            "summary": agg, "per_seed": per_seed}

    a, b = out["arms"]["trainfit"]["summary"], out["arms"]["legacy"]["summary"]
    out["delta_trainfit_minus_legacy"] = {
        k: round(a[k]["mean"] - b[k]["mean"], 6) for k in a if k in b}
    out["n_features_dropped_by_trainfit"] = (
        out["arms"]["legacy"]["n_features"] - out["arms"]["trainfit"]["n_features"])

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"m4_{field}.json", "w"), indent=2)
    d = out["delta_trainfit_minus_legacy"]
    print(f"[T2.3] {field:10} auc_pr {b['auc_pr']['mean']:.4f} -> "
          f"{a['auc_pr']['mean']:.4f} ({d.get('auc_pr', 0):+.4f})   "
          f"auc_roc {b['auc_roc']['mean']:.4f} -> {a['auc_roc']['mean']:.4f} "
          f"({d.get('auc_roc', 0):+.4f})   feats -{out['n_features_dropped_by_trainfit']}")
    return 0


def main():
    if os.environ.get("R19_FIELD"):
        return run_one(os.environ["R19_FIELD"])
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        env = dict(os.environ)
        env.update(R19_FIELD=f, DATASET=f,
                   DATASET_PATH=str(ROOT / "data" / f"clean_dataset_{f}.parquet"))
        env.pop("NEURO_DATASET", None)
        subprocess.call([sys.executable, str(Path(__file__).resolve())], env=env)

    rows = {f: json.load(open(OUT / f"m4_{f}.json")) for f in FIELDS
            if (OUT / f"m4_{f}.json").exists()}
    summary = {
        "task": "T2.3", "finding": "W5",
        "change": ("TF-IDF vectorizer fitted on the training split only; "
                   "transform applied to all rows, out-of-vocabulary tokens dropped"),
        "legacy_path_retained_as": 'build_features(concepts="tfidf_legacy")',
        "fields": {f: {"legacy": r["arms"]["legacy"]["summary"],
                       "trainfit": r["arms"]["trainfit"]["summary"],
                       "delta": r["delta_trainfit_minus_legacy"],
                       "n_features_dropped": r["n_features_dropped_by_trainfit"]}
                   for f, r in rows.items()}}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n[T2.3] -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
