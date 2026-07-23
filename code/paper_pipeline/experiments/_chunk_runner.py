"""Internal helper: run E1/E9a seed-chunks that fit small execution windows.
Appends per-seed rows to results_econ/*.jsonl; aggregate afterwards with _aggregate.py.
Usage: python _chunk_runner.py e1 0 1     # experiment, seeds...
       python _chunk_runner.py e9a 3 4
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D

_CACHE = {}


def get_split():
    if "split" not in _CACHE:
        _CACHE["split"] = D.temporal_split(D.load_dataset())
    return _CACHE["split"]


def main():
    exp, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    df_split = get_split()
    out_path = C.RESULTS_DIR / f"{exp}_perseed.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            done.add(json.loads(line)["seed"])

    with open(out_path, "a") as f:
        for seed in seeds:
            if seed in done:
                print(f"[{exp}] seed {seed} already done"); continue
            if exp == "e1":
                from e1_baselines import run
                row = {"seed": seed, "models": run(seed, df_split, _CACHE)}
            elif exp == "e9a":
                import numpy as np
                from e9a_placebo import shuffle_within_cohort
                from sklearn.ensemble import HistGradientBoostingClassifier
                from utils import stats as S
                if "X" not in _CACHE:
                    _CACHE["X"], _ = D.build_features(df_split, concepts="none")
                sh = shuffle_within_cohort(df_split, seed)
                (Xtr, ytr_sh) = D.split_xy(sh, _CACHE["X"])["train"]
                (Xte, yte) = D.split_xy(df_split, _CACHE["X"])["test"]
                m = HistGradientBoostingClassifier(random_state=seed, max_iter=300,
                                                   early_stopping=True, validation_fraction=0.15)
                m.fit(Xtr, ytr_sh)
                row = {"seed": seed, "metrics": S.evaluate(yte, m.predict_proba(Xte)[:, 1])}
            else:
                sys.exit(f"unknown experiment {exp}")
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"[{exp}] seed {seed} done")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
