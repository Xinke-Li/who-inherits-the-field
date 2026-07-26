#!/usr/bin/env python3
"""T3.4: TabPFN on GPU, without the 1,000-row CPU context handicap.

r15_tabpfn_standalone ran TabPFN v2 on CPU, where the enforced context is 1,000
samples, so every discipline trained on a seeded 1,000-row subsample against
full-train baselines. B8 recorded that as a handicap and named a full-context
GPU run as rebuttal material. This is that run.

Held fixed against r15: the same eight features (config.TABULAR_FEATURES plus
the two boolean coauthorship columns), the same temporal split, the same ten
seeds, the same eq. (2) evaluation against M5 prime that r15f_tabpfn_eq2
applies. Changed: device cuda, and the training context is the whole training
split rather than 1,000 rows.

Two disclosures the output carries and the paper must repeat.

Neuroscience and chemistry train on 14,225 and 16,378 rows, above the 10,000-row
ceiling TabPFN v2 was pretrained for, so those two cells run with
ignore_pretraining_limits set. That is a documented extrapolation of the model,
not a tuned setting, and it is recorded per cell in
`exceeds_pretraining_limit`.

The CPU run drew a fresh 1,000-row subsample per seed, so its ten seeds varied
through the subsample. With the full context there is no subsample, and seed
variation comes from TabPFN's own ensemble permutations through random_state.
The two seed distributions therefore measure different things, and the GPU run
is not a paired rerun of the CPU one.

Out-of-memory is not handled by quietly shrinking the context. The script exits
with code 13 and names the flag to re-run with, because a silent subsample
would reproduce exactly the handicap this task exists to remove.

  DATASET=chemistry python code/r33_tabpfn_gpu.py --stage all

Output: results/revision/T3_4_tabpfn_gpu/<field>/{scores.json,summary.json}
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

FIELD = os.environ.get("DATASET", "chemistry")
OUT = ROOT / "results" / "revision" / "T3_4_tabpfn_gpu" / FIELD
SEEDS = list(range(10))
N_BOOT = 2000
PRETRAIN_LIMIT = 10_000        # TabPFN v2's documented training-set ceiling

# r15_tabpfn_standalone.FEATURES verbatim: config.TABULAR_FEATURES + BOOL_FEATURES
FEATURES = ["early_overlap", "early_prod", "early_breadth", "adv_early_prod",
            "adv_early_breadth", "adv_career_age_at_t0", "coauth_early_n",
            "coauth_early"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_OOM = 13


def env_provenance():
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    try:
        import tabpfn
        tv = getattr(tabpfn, "__version__", "unknown")
    except Exception:
        tv = None
    return {"gpu": gpu, "cuda": getattr(torch.version, "cuda", None),
            "torch": torch.__version__, "tabpfn": tv,
            "python": platform.python_version(), "platform": platform.platform()}


def stage_fit(max_train):
    from sklearn.metrics import average_precision_score
    from utils import data as D
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "r33: no CUDA device. This task exists to remove TabPFN's CPU "
            "context limit; running it on CPU would reproduce the handicap "
            "under a GPU label. Refusing.")
    from tabpfn import TabPFNClassifier

    ds = D.temporal_split(D.load_dataset())
    X = ds[FEATURES].astype(float).values
    y = ds.y.values.astype(int)
    tr = (ds.split == "train").values
    te = (ds.split == "test").values
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    te_pid = ds.loc[te, "student_pid"].astype(str).tolist()

    n_avail = int(len(ytr))
    if max_train and max_train < n_avail:
        print(f"[r33] WARNING: --max-train {max_train} caps the context below "
              f"the {n_avail} available training rows. This is the handicap "
              f"T3.4 exists to remove and is recorded in the output.")
    n_used = min(max_train, n_avail) if max_train else n_avail
    over = n_used > PRETRAIN_LIMIT

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scores.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    scores = rec.get("test_scores_per_seed", {})

    for seed in SEEDS:
        if str(seed) in scores:
            print(f"[r33] {FIELD} seed {seed} already done, skipping")
            continue
        if n_used < n_avail:
            idx = np.random.default_rng(seed).choice(n_avail, n_used, replace=False)
        else:
            idx = np.arange(n_avail)
        t = time.time()
        try:
            m = TabPFNClassifier(device="cuda", random_state=seed,
                                 ignore_pretraining_limits=bool(over))
            m.fit(Xtr[idx], ytr[idx])
            p = m.predict_proba(Xte)[:, 1]
        except torch.cuda.OutOfMemoryError:
            print(f"\n[r33] CUDA OUT OF MEMORY at {n_used} training rows.")
            print("  Not subsampling silently: that is the CPU handicap this "
                  "task removes. Re-run this field with an explicit cap, e.g.")
            print(f"    DATASET={FIELD} python code/r33_tabpfn_gpu.py "
                  f"--stage fit --max-train 8000")
            print("  The cap is then recorded in the output as a handicap.")
            return EXIT_OOM
        scores[str(seed)] = [round(float(v), 6) for v in p]
        rec.update({
            "field": FIELD, "model": "tabpfn_v2_gpu_full_context",
            "features": FEATURES,
            "train_rows_available": n_avail, "train_rows_used": int(n_used),
            "context_capped": bool(n_used < n_avail),
            "exceeds_pretraining_limit": bool(over),
            "pretraining_limit": PRETRAIN_LIMIT,
            "seed_variation": ("TabPFN ensemble permutations via random_state; "
                               "no training subsample, unlike the CPU run"),
            "test_student_pid": te_pid, "test_labels": [int(v) for v in yte],
            "test_scores_per_seed": scores, "env": env_provenance(),
        })
        path.write_text(json.dumps(rec, indent=2))
        print(f"[r33] {FIELD} seed {seed}: AUC-PR "
              f"{average_precision_score(yte, p):.4f} "
              f"({n_used} train rows, {time.time()-t:.0f}s)", flush=True)
    return EXIT_OK


def stage_eval():
    from scipy.stats import wilcoxon
    from e12_corrected_aggregation import fast_auc_pr, fit_val_symmetric
    from utils import data as D
    from utils import stats as S

    rec = json.loads((OUT / "scores.json").read_text())
    ds = D.temporal_split(D.load_dataset())
    Xt, _ = D.build_features(ds, concepts="none")
    nfa = D.build_nfa_features(ds)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    p5 = D.split_xy(ds, X5)
    (Xtr5, ytr), (Xva5, yva), (Xte5, yte) = p5["train"], p5["val"], p5["test"]
    te_pid = ds.loc[ds.split == "test", "student_pid"].astype(str).tolist()

    if te_pid != rec["test_student_pid"]:
        raise SystemExit(f"r33: {FIELD} TabPFN test rows do not align with the "
                         f"local split by student_pid.")
    if list(yte) != rec["test_labels"]:
        raise SystemExit(f"r33: {FIELD} label mismatch between the fit stage "
                         f"and the evaluation split.")

    tab = {s: np.array(rec["test_scores_per_seed"][str(s)], float) for s in SEEDS}
    tab_ap = [fast_auc_pr(yte, tab[s]) for s in SEEDS]

    res_dir = Path(os.environ.get("RESULTS_DIR",
                                  ROOT / "results" / f"results_{FIELD}"))
    m5p_scores, m5p_ap = {}, []
    for s in SEEDS:
        p_te, _, _, _ = fit_val_symmetric(Xtr5, ytr, Xva5, yva, Xte5, s)
        m5p_scores[s] = p_te
        m5p_ap.append(fast_auc_pr(yte, p_te))
    stored = json.loads((res_dir / "e12_corrected_vs_m5.json").read_text())
    m5p_stored = stored["ceilings"]["per_seed"]["M5_prime_val_symmetric"]
    dev = max(abs(a - b) for a, b in zip(m5p_ap, m5p_stored))
    if dev >= 5e-4:
        raise SystemExit(f"r33: {FIELD} M5' refit deviates from the stored e12 "
                         f"ceiling by {dev:.6f}. Refusing to report a verdict "
                         f"against a ceiling that does not reproduce.")
    print(f"[r33] M5' refit matches stored e12 (max per-seed deviation {dev:.2e})")

    rng = np.random.default_rng(0)
    n = len(yte)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    pooled = np.full(N_BOOT, np.nan)
    for b in range(N_BOOT):
        i = idx[b]; yb = yte[i]
        if yb.sum() in (0, len(yb)):
            continue
        pooled[b] = float(np.mean([fast_auc_pr(yb, tab[s][i])
                                   - fast_auc_pr(yb, m5p_scores[s][i])
                                   for s in SEEDS]))
    ci = [round(float(np.nanpercentile(pooled, 2.5)), 4),
          round(float(np.nanpercentile(pooled, 97.5)), 4)]

    d = float(np.mean(tab_ap) - np.mean(m5p_ap))
    p_raw = float(wilcoxon(np.array(tab_ap) - np.array(m5p_ap)).pvalue)
    p_bh = float(S.bh_correction([p_raw])[0][0])
    gates = {"seed_mean_gt_ceiling": bool(np.mean(tab_ap) > np.mean(m5p_ap)),
             "p_adj_lt_0.05": bool(p_bh < 0.05),
             "student_ci_lower_gt_0": bool(ci[0] > 0)}

    cpu = ROOT / "results" / "robustness" / "extra_rungs_partial" / f"{FIELD}_tabpfn.json"
    cpu_mean = json.loads(cpu.read_text())["mean"] if cpu.exists() else None

    out = {
        "experiment": "T3_4_tabpfn_gpu", "field": FIELD,
        "target_table": "Table 25", "target_row": "TabPFN (GPU, full context)",
        "model": rec["model"], "features": rec["features"],
        "train_rows_available": rec["train_rows_available"],
        "train_rows_used": rec["train_rows_used"],
        "context_capped": rec["context_capped"],
        "exceeds_pretraining_limit": rec["exceeds_pretraining_limit"],
        "seed_variation": rec["seed_variation"],
        "tabpfn_mean_auc_pr": round(float(np.mean(tab_ap)), 4),
        "per_seed_auc_pr": [round(v, 4) for v in tab_ap],
        "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
        "delta_vs_M5prime": round(d, 4),
        "wilcoxon_p_raw": round(p_raw, 6), "p_BH": round(p_bh, 6),
        "student_ci95_vs_M5prime": ci,
        "gates_fair": gates, "exceeds_fair": all(gates.values()),
        "cpu_1000row_mean_auc_pr": cpu_mean,
        "gpu_minus_cpu": (round(float(np.mean(tab_ap)) - cpu_mean, 4)
                          if cpu_mean is not None else None),
        "comparison_note": ("the CPU rung varied its ten seeds through a fresh "
                            "1,000-row training subsample; this run has no "
                            "subsample and varies through TabPFN's ensemble "
                            "permutations, so the two are not paired"),
        "env": rec["env"],
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 72)
    print(f"T3.4 {FIELD} TabPFN GPU, {rec['train_rows_used']} training rows")
    print(f"  mean AUC-PR {out['tabpfn_mean_auc_pr']:.4f}  vs M5' "
          f"{out['M5prime_mean']:.4f}  delta {d:+.4f}  CI {ci}  "
          f"p_BH {p_bh:.4f}  exceeds={out['exceeds_fair']}")
    if cpu_mean is not None:
        print(f"  CPU 1,000-row rung {cpu_mean:.4f}, GPU minus CPU "
              f"{out['gpu_minus_cpu']:+.4f} (unpaired, see comparison_note)")
    print("=" * 72)
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["fit", "eval", "all"])
    ap.add_argument("--max-train", type=int, default=0,
                    help="0 means the full training split, which is the point "
                         "of this task. A positive value is recorded as a cap.")
    args = ap.parse_args()
    if args.stage in ("fit", "all"):
        rc = stage_fit(args.max_train)
        if rc != EXIT_OK:
            return rc
    if args.stage in ("eval", "all"):
        return stage_eval()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
