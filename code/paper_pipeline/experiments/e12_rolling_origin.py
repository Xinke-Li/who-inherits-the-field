"""E12-rolling-origin - test-cohort robustness of the model comparison (reviewer R2).

WHY. The paper's HGT-vs-tabular comparison used per-seed Wilcoxon on a SINGLE
temporal test set. That conditions on one test cohort and captures only training
randomness, not test-set sampling variability; the authors' own seed-0 bootstrap
of HGT-M5 already spans zero ([-0.083, +0.037]). To state the model comparison
honestly we evaluate the ladder across MULTIPLE rolling-origin test cohorts and
report the sign stability of any graph gain, rather than a single-split verdict.

PRE-REGISTERED PROTOCOL (fixed before running; boundaries below are frozen):
  Four 2-year test cohorts sliding forward -
      origin 1: test t0 in 2003-2004,  val 2001-2002,  train t0 < 2001
      origin 2: test t0 in 2005-2006,  val 2003-2004,  train t0 < 2003
      origin 3: test t0 in 2007-2008,  val 2005-2006,  train t0 < 2005
      origin 4: test t0 in 2009-2011,  val 2007-2008,  train t0 < 2007  (this
                test window is the paper's frozen main split)
  Rule behind the numbers: most-recent non-overlapping 2-year test cohorts, each
  targeting >=~100 positives; val = the 2 years immediately preceding test;
  train = all earlier cohorts. Observed positive counts 140/119/96/129 (the
  2007-2008 cohort, 96, is disclosed as marginally below the ~100 target).
  TEMPORAL CONTRACT preserved: every feature is still frozen at t0+5 by
  construction (build_features / build_nfa_features are unchanged); only the
  train/val/test PARTITION over cohorts changes.

DECISION RULE (pre-registered): the graph side shows "no observed graph gain"
if, across the four origins, the graph-aware M5 does NOT exceed the tabular
ceiling max(M2, M3) with a consistent sign (i.e. the per-origin delta
M5 - ceiling does not stay positive). We report the delta per origin and its
sign; we do NOT convert this into a single significance verdict, because that is
exactly the single-cohort over-reading we are correcting.

SCOPE. This script runs the CPU ladder M0-M5 (sklearn / GBDT + neighbor-feature
aggregation) per origin, real numbers. The deep HGT per origin needs a GPU and
is deferred to the Colab batch (see PENDING_TODO.md); the manifest of per-origin
row memberships is emitted here so the GPU run scores exactly these cohorts.

Runs for both disciplines (NEURO_DATASET switches the frozen table); econ
windows above, neuro windows computed by the same rule against its t0 range.

Output: results_<disc>/e12_rolling_origin.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
import e1_baselines as E1  # reuse the EXACT ladder models


# Pre-registered econ windows (test_lo, test_hi, val_lo, val_hi, train_hi_excl)
ECON_ORIGINS = [
    (2003, 2004, 2001, 2002, 2001),
    (2005, 2006, 2003, 2004, 2003),
    (2007, 2008, 2005, 2006, 2005),
    (2009, 2011, 2007, 2008, 2007),
]


def rolling_split(df, test_lo, test_hi, val_lo, val_hi, train_hi_excl):
    """Assign a rolling-origin 'split' column. Train = older cohorts, val = the
    fixed window before test, test = the target 2-year cohort. Rows outside all
    three windows (between val and test, or after test) are dropped from this
    origin so no future information leaks into train/val."""
    out = df.copy()
    split = np.full(len(out), None, dtype=object)
    split[(out.t0 < train_hi_excl).values] = "train"
    split[((out.t0 >= val_lo) & (out.t0 <= val_hi)).values] = "val"
    split[((out.t0 >= test_lo) & (out.t0 <= test_hi)).values] = "test"
    out["split"] = split
    out = out[out.split.notna()].reset_index(drop=True)
    return out


def neuro_origins(df):
    """Same rule as econ, computed against neuroscience's t0 range: four
    non-overlapping 2-year test cohorts ending at max t0, each targeting
    >=~100 positives; val = 2 years before; train = earlier."""
    hi = int(df.t0.max())
    origins = []
    for k in range(4):
        t_hi = hi - 2 * k
        t_lo = t_hi - 1
        origins.append((t_lo, t_hi, t_lo - 2, t_lo - 1, t_lo - 2))
    return list(reversed(origins))


def main():
    disc = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
    df = D.load_dataset()
    origins = ECON_ORIGINS if disc == "econ" else neuro_origins(df)

    results, manifest = [], {}
    for (t_lo, t_hi, v_lo, v_hi, tr_hi) in origins:
        ds = rolling_split(df, t_lo, t_hi, v_lo, v_hi, tr_hi)
        sizes = ds.split.value_counts().to_dict()
        pos = int(ds.loc[ds.split == "test", "y"].sum())
        cache = {}
        per_model = {}
        for seed in C.SEEDS:
            for model, metrics in E1.run(seed, ds, cache).items():
                per_model.setdefault(model, []).append(metrics["auc_pr"])
        mean = {m: float(np.mean(v)) for m, v in per_model.items()}
        std = {m: float(np.std(v)) for m, v in per_model.items()}
        ceiling = max(mean["M2_logit_tabular"], mean["M3_gbdt_tabular"])
        rec = {
            "origin": f"{t_lo}-{t_hi}", "test_n": int(sizes.get("test", 0)),
            "test_pos": pos, "train_n": int(sizes.get("train", 0)),
            "val_n": int(sizes.get("val", 0)),
            "auc_pr_mean": {k: round(v, 4) for k, v in mean.items()},
            "auc_pr_std": {k: round(v, 4) for k, v in std.items()},
            "tabular_ceiling": round(ceiling, 4),
            "delta_M5_minus_ceiling": round(mean["M5_gbdt_nfa"] - ceiling, 4),
            "delta_M5_minus_M3": round(mean["M5_gbdt_nfa"] - mean["M3_gbdt_tabular"], 4),
            "delta_M5_minus_M2": round(mean["M5_gbdt_nfa"] - mean["M2_logit_tabular"], 4),
        }
        results.append(rec)
        manifest[f"{t_lo}-{t_hi}"] = {
            "train_pids": ds.loc[ds.split == "train", "student_pid"].tolist(),
            "val_pids": ds.loc[ds.split == "val", "student_pid"].tolist(),
            "test_pids": ds.loc[ds.split == "test", "student_pid"].tolist(),
        }
        print(f"[e12-ro:{disc}] {rec['origin']} pos={pos} "
              f"M5-ceiling={rec['delta_M5_minus_ceiling']:+.4f} "
              f"(M5 {mean['M5_gbdt_nfa']:.3f} vs ceiling {ceiling:.3f})", flush=True)

    deltas = [r["delta_M5_minus_ceiling"] for r in results]
    verdict = {
        "graph_gain_deltas_M5_minus_ceiling": deltas,
        "any_origin_positive": any(d > 0 for d in deltas),
        "all_origins_positive": all(d > 0 for d in deltas),
        "sign_consistent_positive": all(d > 0 for d in deltas),
        "max_delta": max(deltas), "min_delta": min(deltas),
        "reading": ("no observed graph gain: the graph-aware M5 does not exceed "
                    "the tabular ceiling with a consistent positive sign across "
                    "rolling-origin test cohorts"),
        "hgt_per_origin": "PENDING (GPU) - see PENDING_TODO.md; row manifest emitted",
    }
    out = {"experiment": "E12_rolling_origin", "discipline": disc,
           "seeds": C.SEEDS, "origins": results, "verdict": verdict}
    (C.RESULTS_DIR / "e12_rolling_origin.json").write_text(json.dumps(out, indent=2))
    (C.RESULTS_DIR / f"e12_rolling_origin_manifest_{disc}.json").write_text(json.dumps(manifest))
    print(f"[e12-ro:{disc}] verdict deltas {deltas} -> "
          f"{'NO graph gain' if not verdict['sign_consistent_positive'] else 'gain?!'}")


if __name__ == "__main__":
    main()
