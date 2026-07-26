#!/usr/bin/env python3
"""T2.1: the graph arm under the strict temporal contract.

Identical to r3_rgcn_symmetric in protocol -- the same 16-configuration grid
(lr x hidden x layers x dropout), selection on seeds 0 and 1 by validation
AUC-PR only, the winner re-run at all ten seeds, and the unified budget of 200
epochs, patience 15, weight decay 1e-4 -- and different in exactly two places:

  * the graph comes from build_graph_v2(contract="strict"), which keys advisor
    nodes by (advisor_pid, t0) and adds prior-cohort sibling edges, closing
    audit findings F1a and F1b;
  * the training loss's class weight is computed from the training split alone,
    closing F2.

Everything else is held fixed so the result is like-for-like against the legacy
readings of +0.022 (pre-registered) and +0.035 (post hoc symmetric).

GATE 1 runs first and blocks. M5 and M5' are reproduced and compared to the
frozen values before any graph model trains. On a mismatch the script stops
rather than proceed to a verdict, because a silently weakened ceiling would
corrupt the one verdict the paper turns on.

Per-seed test scores and labels are persisted (audit finding F8a), so any verdict
here can be re-tested against a different ceiling or uncertainty model on CPU.

Runs nothing remote. Reads local inputs, writes local outputs.

  DATASET=chemistry python code/r25_strict_contract.py --stage all
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
OUT = ROOT / "results" / "revision" / "T2_1_strict_contract" / FIELD
SEEDS = list(range(10))
SEL_SEEDS = [0, 1]
BUDGET = "hgt"   # r3.BUDGETS["hgt"]: 200 epochs, patience 15, wd 1e-4,
                 # class_weight_scope "train" -- the unified budget and F2 fix
N_BOOT = 2000
# frozen comparators, paper Table 3 (M5) and e12_corrected (M5')
# M5 from paper Table 3; M5' means from each discipline's e12_corrected_vs_m5.json
FROZEN = {"chemistry": {"M5": 0.531, "M5p": 0.5374},
          "econ":      {"M5": 0.339, "M5p": 0.3613},
          "math":      {"M5": 0.398, "M5p": 0.4357},
          "neuro":     {"M5": 0.423, "M5p": 0.4249},
          "physics":   {"M5": 0.644, "M5p": 0.6485}}
LEGACY = {"pre_registered_vs_M5p": 0.022, "post_hoc_symmetric_vs_M5p": 0.035}

# --protocol tuned serves two different tables depending on architecture, so the
# destination is declared per cell rather than inferred from the cell name when
# the tables are assembled.
TABLE_MAP = {
    ("hgt", "prereg"):  ("Table 12b", "HGT"),
    ("hgt", "tuned"):   ("Table 12b", "HGT tuned"),
    ("rgcn", "prereg"): ("Table 12b", "RGCN"),
    ("gat", "prereg"):  ("Table 12b", "GAT"),
    ("rgcn", "tuned"):  ("Table 13b", "RGCN symmetric"),
    ("gat", "tuned"):   ("Table 13b", "GAT symmetric"),
}

# Explicit exit codes. A caller must be able to tell "deliberately did nothing"
# from "failed", which a bare non-zero cannot express.
EXIT_OK = 0            # the stage ran
EXIT_ERROR = 1         # a real failure
EXIT_GATE1_FAILED = 2  # the frozen ceiling did not reproduce; nothing ran
EXIT_SKIPPED = 10      # intentionally nothing to do, with a printed reason
EXIT_NO_CONFIG = 11    # a required configuration file is absent from this
                       # environment; distinct from a real failure so a
                       # bundle-only run can be told apart from a bug

# Stage 1, the pre-registered protocol: fixed config, 300 epochs, patience 30,
# no weight decay. The class weight is train-only here, which is the F2 fix and
# the only intended difference from the frozen pre-registered run.
PREREG_HP = {"lr": 1e-3, "hidden": 64, "layers": 2, "dropout": 0.0}
ARCHS = ("rgcn", "gat", "hgt")


def legacy_hp(field, arch):
    """B(i): the configuration the legacy grid selected for this cell.

    Returns (hp, path). hp is None when the file is absent, and the caller
    reports the exact path it looked for -- this file lives under
    results/robustness/, which must be packed into the bundle explicitly.
    """
    p = (ROOT / "results" / "robustness" / "full_symmetric_grid" /
         f"{field}_{arch}_sym_seed0.json")
    if p.exists():
        return json.loads(p.read_text()).get("hp"), p
    return None, p


def env_provenance():
    import torch
    try:
        import torch_geometric as pyg
        pyg_v = pyg.__version__
    except Exception:
        pyg_v = None
    gpu, drv = None, None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        drv = getattr(torch.version, "cuda", None)
    return {"gpu": gpu, "cuda": drv, "torch": torch.__version__,
            "torch_geometric": pyg_v, "python": platform.python_version(),
            "platform": platform.platform(),
            "note": ("exact cross-GPU reproduction is not assumed; see "
                     "DETERMINISM_NOTE.json in the symmetric grid")}


def gate1(df_split, D, E12, res_dir):
    """Reproduce M5 and M5' before anything trains. Blocks on mismatch."""
    from sklearn.metrics import average_precision_score
    e1 = json.loads((res_dir / "e1_baselines.json").read_text())
    m5_ap = [r["auc_pr"] for r in e1["per_seed"]["M5_gbdt_nfa"]]

    Xt, _ = D.build_features(df_split, concepts="none")
    nfa = D.build_nfa_features(df_split)
    X5 = np.hstack([Xt, nfa.values.astype(float)])
    p5 = D.split_xy(df_split, X5)
    (Xtr5, ytr), (Xva5, yva), (Xte5, yte) = p5["train"], p5["val"], p5["test"]

    stored = json.loads((res_dir / "e12_corrected_vs_m5.json").read_text())
    m5p_stored = stored["ceilings"]["per_seed"]["M5_prime_val_symmetric"]

    m5p_scores, m5p_ap, devs = {}, [], []
    for s in SEEDS:
        p_te, _, _, _ = E12.fit_val_symmetric(Xtr5, ytr, Xva5, yva, Xte5, s)
        m5p_scores[s] = p_te
        ap = float(average_precision_score(yte, p_te))
        m5p_ap.append(ap)
        devs.append(abs(ap - m5p_stored[s]))

    fz = FROZEN.get(FIELD, {})
    g = {
        "M5_mean": round(float(np.mean(m5_ap)), 4),
        "M5_frozen": fz.get("M5"),
        "M5_delta": (round(float(np.mean(m5_ap)) - fz["M5"], 4)
                     if fz.get("M5") is not None else None),
        "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
        "M5prime_frozen": fz.get("M5p"),
        "M5prime_delta": (round(float(np.mean(m5p_ap)) - fz["M5p"], 4)
                          if fz.get("M5p") is not None else None),
        "M5prime_max_per_seed_deviation": round(float(max(devs)), 6),
    }
    ok = True
    for k in ("M5_delta", "M5prime_delta"):
        if g[k] is not None and abs(g[k]) >= 5e-4:
            ok = False
    if g["M5prime_max_per_seed_deviation"] >= 5e-4:
        ok = False
    g["passes"] = ok

    print("=" * 72)
    print(f"GATE 1  harness validation, {FIELD}")
    print(f"  M5   reproduced {g['M5_mean']:.4f}  frozen {g['M5_frozen']}  "
          f"delta {g['M5_delta']}")
    print(f"  M5'  reproduced {g['M5prime_mean']:.4f}  frozen {g['M5prime_frozen']}  "
          f"delta {g['M5prime_delta']}")
    print(f"  M5'  max per-seed deviation {g['M5prime_max_per_seed_deviation']:.6f}")
    print(f"  GATE 1: {'PASS' if ok else 'FAIL'}")
    print("=" * 72)
    if not ok:
        print("\nSTOPPING. The ceiling does not reproduce, so no verdict from this "
              "run would be trustworthy. Fix the harness before training anything.")
    return g, ok, (m5_ap, m5p_ap, m5p_scores, yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["gate1", "grid", "seeds", "aggregate", "all"])
    ap.add_argument("--arch", default="rgcn", choices=list(ARCHS))
    ap.add_argument("--protocol", default="tuned", choices=["prereg", "tuned"],
                    help="prereg = stage 1 fixed config, 300/30/no-wd; "
                         "tuned = stages 2 and 3, grid under the unified budget")
    ap.add_argument("--config", default="strict", choices=["strict", "legacy"],
                    help="B: strict re-selects the grid; legacy pins the "
                         "configuration the legacy grid chose, isolating the "
                         "construction effect from the configuration effect")
    ap.add_argument("--smoke", action="store_true",
                    help="1 config, 1 seed, 3 epochs: proves the path executes "
                         "and emits a schema-valid summary. Numbers are meaningless.")
    args = ap.parse_args()
    globals()["ARGS"] = args

    global SEEDS, SEL_SEEDS, BUDGET
    import config as C
    import e12_corrected_aggregation as E12
    import e2_hgt as E2
    import r3_rgcn_symmetric as R3
    from utils import data as D

    cell = f"{args.arch}_{args.protocol}_{args.config}"
    globals()["OUT"] = OUT / cell
    # Table 9: HGT standard and HGT tuned run at 200/15/wd 1e-4; RGCN and
    # GAT+cohort-time run at 300/30/no wd. So "prereg" resolves by architecture,
    # not globally. Resolving it globally would have run every HGT row at RGCN's
    # budget and made Table 12b's HGT rows incomparable to Table 12's.
    # For HGT, prereg and tuned differ only in configuration, fixed versus
    # grid-selected, never in budget. That matches the original.
    R3.BUDGETS["prereg_strict"] = {"epochs": 300, "patience": 30,
                                   "weight_decay": 0.0,
                                   "class_weight_scope": "train"}
    if args.protocol == "prereg":
        BUDGET = "prereg_strict" if args.arch in ("rgcn", "gat") else "hgt"
    if args.smoke:
        SEEDS = [0]
        SEL_SEEDS = [0]
        R3.BUDGETS["smoke"] = {"epochs": 3, "patience": 2, "weight_decay": 1e-4,
                               "class_weight_scope": "train"}
        BUDGET = "smoke"
        globals()["OUT"] = (ROOT / "results" / "revision" /
                            "T2_1_smoke" / FIELD / cell)
    OUT.mkdir(parents=True, exist_ok=True)
    df = D.temporal_split(D.load_dataset())
    # C.RESULTS_DIR resolves inside the package; the frozen results live in the
    # repo tree (or beside the bundle on Colab).
    res_dir = Path(os.environ.get("RESULTS_DIR", ROOT / "results" / f"results_{FIELD}"))

    g1, ok, ceil = gate1(df, D, E12, res_dir)
    (OUT / "gate1.json").write_text(json.dumps(g1, indent=2))
    if not ok:
        return EXIT_GATE1_FAILED
    if args.stage == "gate1":
        return 0

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[t2.1] device={dev} env={json.dumps(env_provenance())}")
    data = E2.build_graph_v2(df, "none", contract="strict")
    if args.arch == "gat":
        data = R3.attach_t0_norm(data, df)   # F6 channel, GAT only
    print(f"[t2.1] strict graph: {len(data.edge_types)} edge types, "
          f"{data['advisor'].num_nodes} advisor nodes")

    import inspect
    if "arch" not in inspect.signature(R3.train_eval).parameters:
        raise SystemExit(
            "r25: train_eval does not accept 'arch'. Refusing to call it "
            "without the parameter, because the callee would silently build "
            "RGCN and the cell would be mislabelled.")

    # RGCNSym and GATSym are distinct named classes, so a name check is real
    # protection there. HGTNet's inner class is literally called "Net", which
    # almost any PyG model would satisfy, so HGT is checked structurally: the
    # architecture is defined by carrying HGTConv layers.
    EXPECT_CLASS = {"rgcn": "RGCNSym", "gat": "GATSym", "hgt": "Net"}

    def _structural_ok(arch, net):
        if arch != "hgt":
            return True
        try:
            from torch_geometric.nn import HGTConv
        except ImportError:
            return False
        return any(isinstance(m, HGTConv) for m in net.modules())

    def fit(seed, hp):
        r = R3.train_eval(data, seed, dev, hp, BUDGET, arch=args.arch)
        want, got = EXPECT_CLASS.get(args.arch), r.get("model_class")
        if want is None:
            raise SystemExit(f"r25: no expected model class recorded for arch "
                             f"{args.arch!r}; refusing to label the output.")
        if got != want:
            raise SystemExit(f"r25: arch {args.arch!r} requested but the model "
                             f"instantiated was {got!r}, expected {want!r}. "
                             f"Refusing to write a mislabelled cell.")
        if not r.get("structural_ok", True):
            raise SystemExit(f"r25: arch {args.arch!r} passed the name check but "
                             f"failed the structural check (no HGTConv layer). "
                             f"Refusing to write a mislabelled cell.")
        return r

    # stage 1 and B(i) both pin the configuration; only stage 2/3 grid-search.
    pinned = None
    if args.protocol == "prereg":
        pinned = PREREG_HP
    elif args.config == "legacy":
        pinned, _legacy_path = legacy_hp(FIELD, args.arch)
        if pinned is None:
            print(f"STOPPING: the legacy configuration file for "
                  f"{FIELD}/{args.arch} is not present in this environment.")
            print(f"  looked for: {_legacy_path}")
            print("  B(i) pins the configuration the legacy grid selected, so "
                  "this file is required. If you are running from the bundle, "
                  "it must be packed under results/robustness/.")
            return EXIT_NO_CONFIG
    if pinned is not None and args.stage == "grid":
        (OUT / "winner.json").write_text(json.dumps(
            {"hp": pinned, "val_auc_pr_mean": None,
             "source": ("pre-registered fixed configuration"
                        if args.protocol == "prereg"
                        else "legacy grid winner, pinned under strict (B i)")},
            indent=2))
        print(f"SKIPPED: the configuration is pinned ({pinned}), so there is no "
              f"grid to search for cell {cell}. winner.json written.")
        return EXIT_SKIPPED

    if pinned is not None:
        (OUT / "winner.json").write_text(json.dumps(
            {"hp": pinned, "val_auc_pr_mean": None,
             "source": ("pre-registered fixed configuration"
                        if args.protocol == "prereg"
                        else "legacy grid winner, pinned under strict (B i)")},
            indent=2))
        print(f"[t2.1] configuration pinned: {pinned}")

    # ---- grid on selection seeds, validation AUC-PR only ----
    grid_f = OUT / "grid.json"
    if pinned is None and (args.stage in ("grid", "all")
                           or not grid_f.exists()):
        grid = json.loads(grid_f.read_text()) if grid_f.exists() else {}
        allhp = list(R3.iter_grid()) if hasattr(R3, "iter_grid") else list(_grid(E2))
        if args.smoke:
            allhp = allhp[:1]
        for hp in allhp:
            tag = R3.hp_tag(hp)
            if tag in grid:
                continue
            vals = []
            for s in SEL_SEEDS:
                r = fit(s, hp)
                vals.append(r["val_auc_pr"])
            grid[tag] = {"hp": hp, "val_auc_pr_mean": float(np.mean(vals))}
            grid_f.write_text(json.dumps(grid, indent=2))
            print(f"[grid] {tag} val {np.mean(vals):.4f}", flush=True)
        best = max(grid.values(), key=lambda v: v["val_auc_pr_mean"])
        (OUT / "winner.json").write_text(json.dumps(best, indent=2))
        print(f"[t2.1] winner {best['hp']} val {best['val_auc_pr_mean']:.4f}")

    if args.stage == "grid":
        return 0

    # ---- winner at ten seeds, checkpointed per seed ----
    best = json.loads((OUT / "winner.json").read_text())
    for s in SEEDS:
        f = OUT / f"seed{s}.json"
        if f.exists():
            print(f"[seeds] seed {s} already done, skipping")
            continue
        t = time.time()
        r = fit(s, best["hp"])
        f.write_text(json.dumps({
            "seed": s, "hp": best["hp"], "budget": BUDGET,
            "contract": "strict", "class_weight": "train_only",
            "model_class": r["model_class"],
            "arch_requested": r["arch_requested"],
            "test_auc_pr": r["auc_pr"], "test_auc_roc": r["auc_roc"],
            "val_auc_pr": r["val_auc_pr"],
            # audit finding F8a: the arrays a re-analysis needs
            "test_scores": [round(float(x), 5) for x in r["test_scores"]],
            "test_labels": [int(v) for v in r["test_labels"]],
            "seconds": round(time.time() - t, 1),
            "env": env_provenance(),
        }, indent=2))
        print(f"[seeds] seed {s} test AUC-PR {r['auc_pr']:.4f} "
              f"({time.time()-t:.0f}s)", flush=True)

    if args.stage == "seeds":
        return 0
    return aggregate(ceil, best)


def _accepts_arch(fn):
    import inspect
    return "arch" in inspect.signature(fn).parameters


def _grid(E2):
    import itertools
    g = E2.HP_GRID
    keys = list(g)
    for combo in itertools.product(*(g[k] for k in keys)):
        yield dict(zip(keys, combo))


ARGS = None


def aggregate(ceil, best):
    from scipy.stats import wilcoxon
    from r_eval_util import fast_auc_pr
    from utils import stats as S

    m5_ap, m5p_ap, m5p_scores, yte = ceil
    per = [json.loads((OUT / f"seed{s}.json").read_text()) for s in SEEDS]
    g_ap = [p["test_auc_pr"] for p in per]
    g_sc = [np.array(p["test_scores"]) for p in per]
    y = np.array(per[0]["test_labels"])

    rng = np.random.default_rng(0)
    n = len(y)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    gb = np.full((len(SEEDS), N_BOOT), np.nan)
    cb = np.full((len(SEEDS), N_BOOT), np.nan)
    for b in range(N_BOOT):
        i = idx[b]; yb = y[i]
        if yb.sum() in (0, len(yb)):
            continue
        for k, s in enumerate(SEEDS):
            gb[k, b] = fast_auc_pr(yb, g_sc[k][i])
            cb[k, b] = fast_auc_pr(yb, np.asarray(m5p_scores[s])[i])
    pooled = np.nanmean(gb - cb, axis=0)
    ci = [round(float(np.nanpercentile(pooled, 2.5)), 4),
          round(float(np.nanpercentile(pooled, 97.5)), 4)]

    d5 = float(np.mean(g_ap) - np.mean(m5_ap))
    d5p = float(np.mean(g_ap) - np.mean(m5p_ap))
    p_raw = float(wilcoxon(np.array(g_ap) - np.array(m5p_ap)).pvalue)
    p_bh = float(S.bh_correction([p_raw])[0][0])
    gates = {"seed_mean_gt_ceiling": bool(np.mean(g_ap) > np.mean(m5p_ap)),
             "p_adj_lt_0.05": bool(p_bh < 0.05),
             "student_ci_lower_gt_0": bool(ci[0] > 0)}
    out = {
        "experiment": "T2_1_strict_contract", "field": FIELD,
        "target_table": TABLE_MAP.get((ARGS.arch, ARGS.protocol), (None, None))[0],
        "target_row": TABLE_MAP.get((ARGS.arch, ARGS.protocol), (None, None))[1],
        "model": f"{ARGS.arch}_strict", "cell": f"{ARGS.arch}_{ARGS.protocol}_{ARGS.config}",
        "protocol": ARGS.protocol, "config_source": ARGS.config,
        "model_class": per[0].get("model_class"),
        "arch_requested": per[0].get("arch_requested"),
        "hp": best["hp"], "budget": BUDGET,
        "changes_vs_r3": ["build_graph_v2 contract=strict (F1a, F1b)",
                          "class weight from training split only (F2)"],
        "seed_mean_auc_pr": round(float(np.mean(g_ap)), 4),
        "M5_mean": round(float(np.mean(m5_ap)), 4),
        "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
        "delta_vs_M5": round(d5, 4),
        "delta_vs_M5prime": round(d5p, 4),
        "wilcoxon_p_raw": round(p_raw, 6), "p_BH": round(p_bh, 6),
        "student_ci95_vs_M5prime": ci,
        "gates_fair": gates, "exceeds_fair": all(gates.values()),
        "per_seed_auc_pr": [round(v, 4) for v in g_ap],
        "legacy_comparison": LEGACY,
        "env": env_provenance(),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    print(f"T2.1 {FIELD} RGCN, strict contract")
    print(f"  legacy pre-registered vs M5'  : +{LEGACY['pre_registered_vs_M5p']:.3f}")
    print(f"  legacy post hoc symmetric     : +{LEGACY['post_hoc_symmetric_vs_M5p']:.3f}")
    print(f"  strict contract vs M5'        : {d5p:+.4f}  CI {ci}  "
          f"p_BH {p_bh:.4f}  exceeds={out['exceeds_fair']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
