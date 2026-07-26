"""Chunked driver for E10 (N seeds per invocation; resumable via jsonl)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e10_advisor_placebo as E

PARTS = C.RESULTS_DIR / "e10_parts"
PARTS.mkdir(exist_ok=True)
variant = sys.argv[1]
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
kind = {"true": None, "cohort": "cohort", "random": "random",
        "field_mean": "field_mean"}[variant]
out = PARTS / f"{variant}.jsonl"
done = set()
if out.exists():
    done = {json.loads(l)["seed"] for l in out.read_text().splitlines() if l.strip()}
todo = [s for s in C.SEEDS if s not in done][:nmax]
if not todo:
    print("all done:", variant); sys.exit(0)

df = D.load_dataset()
df_split = D.temporal_split(df)
for seed in todo:
    ov = (df_split.early_overlap.values if kind is None
          else E.placebo_overlap(df_split, kind, seed))
    r = E.run_variant(df_split, ov, seed)
    row = {"seed": seed, "run": r,
           "feat_mean": round(float(np.mean(ov)), 4),
           "feat_std": round(float(np.std(ov)), 4),
           "corr_true": round(float(np.corrcoef(df_split.early_overlap.values, ov)[0, 1]), 4)}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("seed", seed, "done")
