"""E12 corrected aggregation: the pre-registered comparison, recomputed on the
third (canonical) GPU run's per-seed artifacts. CPU only; no graph model is retrained.

Fixes three evaluation-protocol deviations documented in LEAKAGE_AUDIT_e12.md:

  F3  the Colab aggregation compared against max(M2, M3), dropping M5_gbdt_nfa,
      the comparator the pre-registered decision rule names (e1_baselines.py:13-15).
      Here the PRIMARY comparison is per-seed paired vs M5; the max(M2, M3)
      comparison is kept as SECONDARY for continuity with the naive protocol.
  F4  the Colab bootstrap resampled 10 seed-level gaps, omitting test-set sampling
      error. Here the bootstrap is a paired STUDENT-level resample of the test
      cohort (2000 draws): for every draw the same student indices are applied to
      the graph scores and the ceiling scores of the same seed. Reported per seed
      and pooled (the pooled CI resamples students once per draw and averages the
      per-seed paired differences, so it carries both student noise and seed noise).
  F14 the graph models early-stop on the temporal VAL cohort's AUC-PR; M3/M5
      early-stop on a random 15% slice of TRAIN and never see VAL. Here M5' and
      M3' are retrained per seed with a VAL-symmetric budget: fit
      HistGradientBoostingClassifier(max_iter=500, early_stopping=False), then
      select the boosting-iteration count that maximizes AUC-PR on the temporal
      VAL cohort via staged_predict_proba (sklearn has no external-eval-set early
      stopping, so the selection is done post hoc over the staged predictions;
      predictions at iteration k are identical whether or not later trees exist,
      so no refit is needed). Test scores are then read at the selected iteration.

  F9 is NOT fixed here (it would need a GPU re-run): h_extra_gnns.py trained
     rgcn / gat_cohort_time with EPOCHS=300 / PATIENCE=30 and no weight decay
     (h_extra_gnns.py:56,138) while e2_hgt.py used 200 / 15 with weight_decay=1e-4
     (e2_hgt.py:43,184). Disclosed in the output's caveats.

Three ceilings are reported per discipline:
  M5_preregistered      per-seed M5_gbdt_nfa from e1_baselines.json (the
                        pre-registered decision rule; original training budget)
  M5_prime_val_symmetric  M5 features, VAL-symmetric budget as above (the fair
                        ceiling; primary for the paper's corrected verdict)
  maxM2M3_naive         per-seed max(M2, M3) (the naive Colab ceiling; secondary)

Gates (three, all required for an "exceeds" verdict), evaluated twice:
  preregistered: vs M5   -- seed_mean > ceiling mean, BH-corrected Wilcoxon
                            p_adj < 0.05, pooled student-level CI lower > 0
  fair:          vs M5'  -- same three gates against the VAL-symmetric ceiling
BH correction spans the four graph models within one discipline (the same family
as the naive e12 aggregation).

Inputs  (per field): results/results_<field>/e1_baselines.json
                     results/results_<field>/results_hgt/hgt_none_seed*.json
                     results/results_<field>/results_hgt_grid/hgt_none_seed*_best.json
                     results/results_<field>/results_extra_gnns/{rgcn,tgat}_seed*.json
Output  (per field): results/results_<field>/e12_corrected_vs_m5.json
Summary (--summarize): results/e12_corrected_summary.json + a markdown table.

The naive Colab aggregation (e12_hgt_vs_baselines.json, produced by the same run)
is left untouched and is re-derived here from the per-seed files as a consistency
check; a mismatch aborts the run.

Usage:
  python code/e12_corrected_aggregation.py --field econ
  python code/e12_corrected_aggregation.py --all          # 5 subprocesses
  python code/e12_corrected_aggregation.py --summarize    # after --all
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
N_BOOT = 2000
MODELS = {
    "hgt": "results_hgt/hgt_none_seed{s}.json",
    "hgt_tuned": "results_hgt_grid/hgt_none_seed{s}_best.json",
    "rgcn": "results_extra_gnns/rgcn_seed{s}.json",
    "gat_cohort_time": "results_extra_gnns/tgat_seed{s}.json",
}


def fast_auc_pr(y, s):
    """Average precision, tie-aware, identical to sklearn's
    average_precision_score (sum over distinct thresholds of dR * P)."""
    order = np.argsort(-s, kind="stable")
    y, s = y[order], s[order]
    tp = np.cumsum(y)
    n_pos = tp[-1]
    if n_pos == 0:
        return 0.0
    # threshold boundaries: last index of each run of equal scores
    boundary = np.append(s[1:] != s[:-1], True)
    tp_b = tp[boundary]
    fp_b = np.flatnonzero(boundary) + 1 - tp_b
    precision = tp_b / (tp_b + fp_b)
    recall = tp_b / n_pos
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(d_recall * precision))


def fit_val_symmetric(Xtr, ytr, Xva, yva, Xte, seed):
    """GBDT with the graph models' selection budget: iteration count chosen to
    maximize AUC-PR on the temporal VAL cohort (see module docstring, F14)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                       early_stopping=False)
    m.fit(Xtr, ytr)
    best_ap, best_iter = -1.0, 1
    for it, proba in enumerate(m.staged_predict_proba(Xva), start=1):
        ap = fast_auc_pr(yva, proba[:, 1])
        if ap > best_ap:
            best_ap, best_iter = ap, it
    for it, proba in enumerate(m.staged_predict_proba(Xte), start=1):
        if it == best_iter:
            p_te = proba[:, 1]
            break
    for it, proba in enumerate(m.staged_predict_proba(Xva), start=1):
        if it == best_iter:
            p_va = proba[:, 1]
            break
    return p_te, p_va, best_iter, best_ap


def fit_original_m5(Xtr, ytr, Xte, seed):
    """M5 exactly as e1_baselines.py:90-94 trains it (random internal 15% slice)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                       early_stopping=True, validation_fraction=0.15)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def naive_consistency_check(res_dir, per_seed_graph, e1):
    """Re-derive the Colab cell-6 aggregation from the per-seed files and compare
    with the naive json shipped in the same run's zip. Aborts on mismatch."""
    naive = json.loads((res_dir / "e12_hgt_vs_baselines.json").read_text())
    m2 = [r["auc_pr"] for r in e1["per_seed"]["M2_logit_tabular"]]
    m3 = [r["auc_pr"] for r in e1["per_seed"]["M3_gbdt_tabular"]]
    ceil = [max(a, b) for a, b in zip(m2, m3)]
    assert abs(np.mean(ceil) - naive["tabular_ceiling_mean_max_M2_M3"]) < 5e-5, \
        "naive ceiling mismatch"
    for name, seeds in per_seed_graph.items():
        g = np.array([seeds[s]["auc_pr"] for s in range(10)])
        rec = naive["models"][name]
        assert abs(g.mean() - rec["seed_mean_auc_pr"]) < 5e-5, f"naive mean mismatch {name}"
        assert abs((g - np.array(ceil)).mean() - rec["delta_vs_ceiling_mean"]) < 5e-5, \
            f"naive delta mismatch {name}"
    return naive


def run_field(field):
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
    from utils import data as D
    from utils import stats as S
    from sklearn.metrics import average_precision_score

    res_dir = RESULTS_ROOT / f"results_{field}"
    e1 = json.loads((res_dir / "e1_baselines.json").read_text())
    seeds = e1["seeds"]
    assert seeds == list(range(10))

    # ---- graph per-seed artifacts ----
    per_seed_graph = {}
    for name, patt in MODELS.items():
        per_seed_graph[name] = {}
        for s in seeds:
            d = json.loads((res_dir / patt.format(s=s)).read_text())
            t = d.get("test", d)
            per_seed_graph[name][s] = {
                "auc_pr": t["auc_pr"],
                "scores": np.array(d["test_scores"], float),
                "labels": np.array(d["test_labels"], int),
            }

    naive = naive_consistency_check(res_dir, per_seed_graph, e1)
    print(f"[{field}] naive re-derivation matches the shipped e12_hgt_vs_baselines.json")

    # ---- local split, features, and the bitwise label check ----
    df = D.load_dataset()
    df_split = D.temporal_split(df)
    Xt, _ = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    tab = D.split_xy(df_split, Xt)
    p5 = D.split_xy(df_split, X5)
    (Xtr3, ytr), (Xva3, yva), (Xte3, yte) = tab["train"], tab["val"], tab["test"]
    (Xtr5, _), (Xva5, _), (Xte5, _) = p5["train"], p5["val"], p5["test"]

    for name in MODELS:
        for s in seeds:
            assert (per_seed_graph[name][s]["labels"] == yte).all(), \
                f"{field}/{name}/seed{s}: test labels disagree with the local split"
    print(f"[{field}] test labels of all 40 per-seed files match the local temporal "
          f"split bitwise (n_test={len(yte)})")

    # ---- ceilings ----
    m5_pre = [r["auc_pr"] for r in e1["per_seed"]["M5_gbdt_nfa"]]
    m2 = [r["auc_pr"] for r in e1["per_seed"]["M2_logit_tabular"]]
    m3 = [r["auc_pr"] for r in e1["per_seed"]["M3_gbdt_tabular"]]
    max23 = [max(a, b) for a, b in zip(m2, m3)]

    m5p_scores, m5p_ap, m5p_iters = {}, [], []
    m3p_ap = []
    m5o_scores, m5o_ap = {}, []
    for s in seeds:
        p_te, _, it, _ = fit_val_symmetric(Xtr5, ytr, Xva5, yva, Xte5, s)
        m5p_scores[s] = p_te
        m5p_ap.append(average_precision_score(yte, p_te))
        m5p_iters.append(it)
        p_te3, _, _, _ = fit_val_symmetric(Xtr3, ytr, Xva3, yva, Xte3, s)
        m3p_ap.append(average_precision_score(yte, p_te3))
        m5o_scores[s] = fit_original_m5(Xtr5, ytr, Xte5, s)
        m5o_ap.append(average_precision_score(yte, m5o_scores[s]))
        print(f"[{field}] seed {s}: M5'={m5p_ap[-1]:.4f} (iter {it}) "
              f"M3'={m3p_ap[-1]:.4f} M5_refit={m5o_ap[-1]:.4f} "
              f"(e1 stored {m5_pre[s]:.4f})", flush=True)

    m5_refit_dev = float(np.max(np.abs(np.array(m5o_ap) - np.array(m5_pre))))
    print(f"[{field}] max |M5 refit - e1 stored| = {m5_refit_dev:.2e}")

    # sanity: fast_auc_pr must reproduce sklearn on the full arrays
    for name in MODELS:
        g = per_seed_graph[name][0]
        assert abs(fast_auc_pr(yte, g["scores"])
                   - average_precision_score(yte, g["scores"])) < 1e-12

    # ---- student-level paired bootstrap (F4) ----
    rng = np.random.default_rng(0)
    n = len(yte)
    idx_draws = rng.integers(0, n, size=(N_BOOT, n))
    ceil_ap = {"m5o": np.empty((10, N_BOOT)), "m5p": np.empty((10, N_BOOT))}
    for b in range(N_BOOT):
        idx = idx_draws[b]
        yb = yte[idx]
        for s in seeds:
            ceil_ap["m5o"][s, b] = fast_auc_pr(yb, m5o_scores[s][idx])
            ceil_ap["m5p"][s, b] = fast_auc_pr(yb, m5p_scores[s][idx])

    def model_bootstrap(name):
        gap = np.empty((10, N_BOOT))
        for b in range(N_BOOT):
            idx = idx_draws[b]
            yb = yte[idx]
            for s in seeds:
                gap[s, b] = fast_auc_pr(yb, per_seed_graph[name][s]["scores"][idx])
        out = {}
        for key, tag in (("m5o", "vs_M5"), ("m5p", "vs_M5prime")):
            diff = gap - ceil_ap[key]                       # (10, N_BOOT)
            pooled = diff.mean(axis=0)                      # seed-mean per draw
            out[tag] = {
                "pooled_ci95": [round(float(np.percentile(pooled, 2.5)), 4),
                                round(float(np.percentile(pooled, 97.5)), 4)],
                "per_seed_ci95": [[round(float(np.percentile(diff[s], 2.5)), 4),
                                   round(float(np.percentile(diff[s], 97.5)), 4)]
                                  for s in seeds],
            }
        return out

    # ---- comparisons, BH within the four-model family (as in naive e12) ----
    packs = {name: {} for name in MODELS}
    families = {
        "M5": m5_pre, "M5prime": m5p_ap, "max23": max23,
    }
    pvals = {k: [] for k in families}
    for name in MODELS:
        g = [per_seed_graph[name][s]["auc_pr"] for s in seeds]
        packs[name]["n_seeds"] = 10
        packs[name]["seed_mean_auc_pr"] = round(float(np.mean(g)), 4)
        packs[name]["seed_std_auc_pr"] = round(float(np.std(g, ddof=1)), 4)
        for fam, ceil in families.items():
            t = S.paired_wilcoxon(g, ceil)
            packs[name][f"_raw_{fam}"] = t
            pvals[fam].append(t["p"])
    adj = {fam: S.bh_correction(pv)[0] for fam, pv in pvals.items()}

    out_models = {}
    for i, name in enumerate(MODELS):
        boot = model_bootstrap(name)
        g_mean = packs[name]["seed_mean_auc_pr"]
        m = {
            "n_seeds": 10,
            "seed_mean_auc_pr": g_mean,
            "seed_std_auc_pr": packs[name]["seed_std_auc_pr"],
            "delta_vs_M5": round(float(packs[name]["_raw_M5"]["mean_diff"]), 4),
            "p_raw_M5": packs[name]["_raw_M5"]["p"],
            "p_adj_M5": round(float(adj["M5"][i]), 6),
            "delta_vs_M5prime": round(float(packs[name]["_raw_M5prime"]["mean_diff"]), 4),
            "p_raw_M5prime": packs[name]["_raw_M5prime"]["p"],
            "p_adj_M5prime": round(float(adj["M5prime"][i]), 6),
            "secondary_delta_vs_maxM2M3": round(float(packs[name]["_raw_max23"]["mean_diff"]), 4),
            "secondary_p_raw_maxM2M3": packs[name]["_raw_max23"]["p"],
            "secondary_p_adj_maxM2M3": round(float(adj["max23"][i]), 6),
            "bootstrap": {
                "level": "student",
                "n_boot": N_BOOT,
                "seed_scope": "all 10 seeds; per-seed CIs listed, pooled CI = "
                              "per-draw mean over seeds of the paired difference",
                "vs_M5": boot["vs_M5"],
                "vs_M5prime": boot["vs_M5prime"],
                "ci95": boot["vs_M5prime"]["pooled_ci95"],
            },
        }
        gates_pre = {
            "seed_mean_gt_ceiling": bool(g_mean > np.mean(m5_pre)),
            "p_adj_lt_0.05": bool(m["p_adj_M5"] < 0.05),
            "student_ci_lower_gt_0": bool(boot["vs_M5"]["pooled_ci95"][0] > 0),
        }
        gates_fair = {
            "seed_mean_gt_ceiling": bool(g_mean > np.mean(m5p_ap)),
            "p_adj_lt_0.05": bool(m["p_adj_M5prime"] < 0.05),
            "student_ci_lower_gt_0": bool(boot["vs_M5prime"]["pooled_ci95"][0] > 0),
        }
        m["gates_preregistered"] = gates_pre
        m["gates_fair"] = gates_fair
        m["exceeds_preregistered"] = all(gates_pre.values())
        m["exceeds_fair"] = all(gates_fair.values())
        out_models[name] = m
        print(f"[{field}] {name}: mean {g_mean:.4f} | dM5 {m['delta_vs_M5']:+.4f} "
              f"(p_adj {m['p_adj_M5']:.4f}) | dM5' {m['delta_vs_M5prime']:+.4f} "
              f"(p_adj {m['p_adj_M5prime']:.4f}) CI' {boot['vs_M5prime']['pooled_ci95']} | "
              f"exceeds pre={m['exceeds_preregistered']} fair={m['exceeds_fair']}", flush=True)

    out = {
        "experiment": "E12_corrected_vs_M5",
        "field": field,
        "run": "third (canonical)",
        "seeds": seeds,
        "n_test": int(len(yte)),
        "ceilings": {
            "M5_preregistered": round(float(np.mean(m5_pre)), 4),
            "M5_prime_val_symmetric": round(float(np.mean(m5p_ap)), 4),
            "maxM2M3_naive": round(float(np.mean(max23)), 4),
            "M3_prime_val_symmetric": round(float(np.mean(m3p_ap)), 4),
            "per_seed": {"M5_preregistered": [round(v, 4) for v in m5_pre],
                          "M5_prime_val_symmetric": [round(float(v), 4) for v in m5p_ap],
                          "maxM2M3_naive": [round(v, 4) for v in max23],
                          "M3_prime_val_symmetric": [round(float(v), 4) for v in m3p_ap]},
            "M5_prime_selected_iters": m5p_iters,
            "M5_refit_vs_e1_stored_max_abs_dev": m5_refit_dev,
        },
        "models": out_models,
        "naive_reference": {
            "file": "e12_hgt_vs_baselines.json",
            "consistency": "re-derived from the per-seed files of this run; matches",
            "any_exceeds": naive["any_exceeds"],
            "exceeds": [k for k, v in naive["models"].items() if v.get("exceeds_ceiling")],
        },
        "caveats": [
            "F9: rgcn and gat_cohort_time were trained by h_extra_gnns.py with "
            "EPOCHS=300 / PATIENCE=30 and Adam without weight decay "
            "(h_extra_gnns.py:56,138); hgt and hgt_tuned by e2_hgt.py with "
            "EPOCHS=200 / PATIENCE=15 and weight_decay=1e-4 (e2_hgt.py:43,184). "
            "Same split, same features, same seeds; training budget stated per "
            "architecture. Not equalized here (would need a GPU re-run).",
            "F2: h_extra_gnns.py computes the class weight over train+val+test "
            "labels (h_extra_gnns.py:136-137); e2_hgt.py restricts to train. "
            "Bounded at a 2-4% shift of one loss scalar (audit F2).",
            "F14 symmetrization: M5'/M3' select the boosting-iteration count on "
            "the temporal VAL cohort's AUC-PR via staged_predict_proba (one "
            "selection per candidate iteration, mirroring the graph models' "
            "per-epoch checkpoint selection). The pre-registered M5 numbers keep "
            "the original internal-15%-slice budget.",
            "Bootstrap is student-level and paired; the pooled CI carries both "
            "test-cohort sampling error and seed-to-seed variation.",
        ],
        "any_exceeds_preregistered": any(v["exceeds_preregistered"] for v in out_models.values()),
        "any_exceeds_fair": any(v["exceeds_fair"] for v in out_models.values()),
    }
    path = res_dir / "e12_corrected_vs_m5.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[{field}] wrote {path}")


def summarize():
    rows = []
    for field in FIELDS:
        d = json.loads((RESULTS_ROOT / f"results_{field}" / "e12_corrected_vs_m5.json").read_text())
        naive = json.loads((RESULTS_ROOT / f"results_{field}" / "e12_hgt_vs_baselines.json").read_text())
        for name in MODELS:
            m = d["models"][name]
            rows.append({
                "field": field, "model": name,
                "seed_mean_auc_pr": m["seed_mean_auc_pr"],
                "naive_exceeds": bool(naive["models"][name].get("exceeds_ceiling")),
                "preregistered_exceeds": m["exceeds_preregistered"],
                "fair_exceeds": m["exceeds_fair"],
                "delta_vs_M5": m["delta_vs_M5"],
                "delta_vs_M5prime": m["delta_vs_M5prime"],
                "student_ci95_vs_M5prime": m["bootstrap"]["vs_M5prime"]["pooled_ci95"],
            })
    out = {"experiment": "E12_corrected_summary", "run": "third (canonical)",
           "rows": rows,
           "n_naive": sum(r["naive_exceeds"] for r in rows),
           "n_preregistered": sum(r["preregistered_exceeds"] for r in rows),
           "n_fair": sum(r["fair_exceeds"] for r in rows)}
    (RESULTS_ROOT / "e12_corrected_summary.json").write_text(json.dumps(out, indent=2))
    print("| field | model | naive | preregistered (M5) | fair (M5') |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['field']} | {r['model']} | "
              f"{'EXCEEDS' if r['naive_exceeds'] else 'null'} | "
              f"{'EXCEEDS' if r['preregistered_exceeds'] else 'null'} | "
              f"{'EXCEEDS' if r['fair_exceeds'] else 'null'} |")
    print(f"\nexceeds count: naive {out['n_naive']} -> preregistered "
          f"{out['n_preregistered']} -> fair {out['n_fair']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=FIELDS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return
    if args.all:
        for f in FIELDS:
            env = dict(os.environ)
            env["DATASET"] = f
            env["DATASET_PATH"] = str(ROOT / "data" / f"clean_dataset_{f}.parquet")
            env.pop("NEURO_DATASET", None)
            r = subprocess.run([sys.executable, __file__, "--field", f], env=env)
            if r.returncode:
                sys.exit(f"field {f} failed")
        summarize()
        return
    assert args.field, "--field, --all or --summarize required"
    assert os.environ.get("DATASET") == args.field, \
        "run via --all, or set DATASET / DATASET_PATH env vars for config.py"
    run_field(args.field)


if __name__ == "__main__":
    main()
