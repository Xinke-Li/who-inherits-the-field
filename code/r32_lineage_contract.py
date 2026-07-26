#!/usr/bin/env python3
"""T2.2b: the graph arm under the strict LINEAGE contract.

r25_strict_contract answered "does the chemistry crossing survive a graph that
obeys the time contract". This answers the question underneath reviewer B1:
does the multi-generation genealogy of Figure 2, which enters no model anywhere
in the paper, change any verdict once a model can actually pass messages over
it.

Identical to r25 in every protocol detail, the same 16-configuration grid on
seeds 0 and 1 by validation AUC-PR, the winner at ten seeds, the same budgets
per architecture, the same train-only class weight, the same eq. (2) three
gates against M5 prime. One thing differs: the graph comes from
build_graph_v2(contract="strict_lineage"), which is the strict construction
plus

    advisor --mentored_by--> advisor       the AFT parent of the focal
                                           student's advisor
    advisor --studies_lineage--> concept   that person's top-10 concepts from
                                           works dated at or before the focal
                                           student's t0+5

both keyed to the focal cohort, both with the reverse edge every other relation
gets. Because every strict relation keeps its own tensors bit-for-bit and the
direct-advisor features are standardized on the direct block alone, a
difference between a lineage cell and its matching strict cell is the lineage
relations and nothing else.

Each cell records the matching strict cell's delta and the lineage-minus-strict
difference, so the comparison that isolates lineage is in the artifact rather
than assembled by hand afterwards.

GATE 1 runs first and blocks, exactly as in r25: M5 and M5 prime are reproduced
against the frozen values before anything trains.

Coverage is a real limit and is recorded per cell. The share of rows with a
known grand-advisor runs econ 0.38, math 0.24, neuro 0.58, physics 0.70,
chemistry 0.76; the share whose grand-advisor also has cached works, which is
what the concept channel needs, runs 0.18, 0.08, 0.32, 0.36, 0.47. A null in
mathematics is a null on a graph where three rows in four carry no lineage
edge at all, and must be read that way.

  DATASET=chemistry python code/r32_lineage_contract.py --stage all --arch rgcn
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
sys.path.insert(0, str(ROOT / "code" / "paper_pipeline" / "experiments"))

FIELD = os.environ.get("DATASET", "chemistry")
OUT = ROOT / "results" / "revision" / "T2_2b_lineage_contract" / FIELD
LINEAGE = ROOT / "data" / "supplement" / f"lineage_{FIELD}.parquet"
SEEDS = list(range(10))
SEL_SEEDS = [0, 1]
N_BOOT = 2000

# The strict cells this arm is read against. The assembler resolves the T2.1
# tree; these are the roots a bundle or a full checkout may present it under.
STRICT_ROOTS = [
    ROOT / "results" / "revision" / "T2_1_strict_contract" / "T2_1_final"
    / "T2_1_strict_contract",
    ROOT / "results" / "revision" / "T2_1_strict_contract",
]

TABLE_MAP = {
    ("rgcn", "prereg"): ("Table 12c", "RGCN lineage"),
    ("gat", "prereg"):  ("Table 12c", "GAT lineage"),
    ("rgcn", "tuned"):  ("Table 13c", "RGCN symmetric lineage"),
    ("gat", "tuned"):   ("Table 13c", "GAT symmetric lineage"),
}

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_GATE1_FAILED = 2
EXIT_SKIPPED = 10
EXIT_NO_INPUT = 12       # the lineage table is absent; distinct from a bug

ARCHS = ("rgcn", "gat")
ARGS = None


def strict_summary(arch, protocol):
    """The matching r25 cell, so the lineage delta is reported against it."""
    rel = Path(FIELD) / f"{arch}_{protocol}_strict" / "summary.json"
    tried = []
    for root in STRICT_ROOTS:
        p = root / rel
        tried.append(str(p))
        if p.exists():
            d = json.loads(p.read_text())
            return {"found": True, "path": str(p),
                    "cell": d.get("cell"),
                    "seed_mean_auc_pr": d.get("seed_mean_auc_pr"),
                    "delta_vs_M5prime": d.get("delta_vs_M5prime"),
                    "student_ci95_vs_M5prime": d.get("student_ci95_vs_M5prime"),
                    "p_BH": d.get("p_BH"), "exceeds_fair": d.get("exceeds_fair")}
    return {"found": False, "searched": tried}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["gate1", "grid", "seeds", "aggregate", "all"])
    ap.add_argument("--arch", default="rgcn", choices=list(ARCHS))
    ap.add_argument("--protocol", default="tuned", choices=["prereg", "tuned"])
    ap.add_argument("--smoke", action="store_true",
                    help="1 config, 1 seed, 3 epochs: proves the path executes "
                         "and emits a schema-valid summary. Numbers are "
                         "meaningless and the output goes to a smoke tree.")
    args = ap.parse_args()
    globals()["ARGS"] = args

    global SEEDS, SEL_SEEDS
    import pandas as pd
    import e12_corrected_aggregation as E12
    import e2_hgt as E2
    import r3_rgcn_symmetric as R3
    import r25_strict_contract as R25
    from utils import data as D

    if not LINEAGE.exists():
        print(f"STOPPING: the lineage table is absent.\n  looked for: {LINEAGE}")
        print("  Build it with: python code/r31_lineage_table.py")
        print("  Refusing to fall back to the strict graph, which would be "
              "reported as a lineage null.")
        return EXIT_NO_INPUT

    cell = f"{args.arch}_{args.protocol}_lineage"
    globals()["OUT"] = OUT / cell

    R3.BUDGETS["prereg_strict"] = {"epochs": 300, "patience": 30,
                                   "weight_decay": 0.0,
                                   "class_weight_scope": "train"}
    budget = "hgt"
    if args.protocol == "prereg":
        # Table 9's budgets: RGCN and GAT+cohort-time ran 300/30/no weight
        # decay. Both architectures here are in that group, so unlike r25 there
        # is no per-architecture branch, but the value is the same one r25
        # resolves to for these two.
        budget = "prereg_strict"
    if args.smoke:
        SEEDS = [0]
        SEL_SEEDS = [0]
        R3.BUDGETS["smoke"] = {"epochs": 3, "patience": 2, "weight_decay": 1e-4,
                               "class_weight_scope": "train"}
        budget = "smoke"
        globals()["OUT"] = (ROOT / "results" / "revision" /
                            "T2_2b_smoke" / FIELD / cell)
    OUT.mkdir(parents=True, exist_ok=True)

    df = D.temporal_split(D.load_dataset())
    res_dir = Path(os.environ.get("RESULTS_DIR",
                                  ROOT / "results" / f"results_{FIELD}"))

    g1, ok, ceil = R25.gate1(df, D, E12, res_dir)
    (OUT / "gate1.json").write_text(json.dumps(g1, indent=2))
    if not ok:
        return EXIT_GATE1_FAILED
    if args.stage == "gate1":
        return EXIT_OK

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lin = pd.read_parquet(LINEAGE)
    data = E2.build_graph_v2(df, "none", contract="strict_lineage", lineage=lin)
    if args.arch == "gat":
        data = R3.attach_t0_norm(data, df)
    lstats = dict(getattr(data, "lineage_stats", {}))
    if not lstats:
        raise SystemExit(
            "r32: the graph carries no lineage_stats, so it is not a lineage "
            "graph. Refusing to write a cell labelled as one.")
    for rel in (("advisor", "mentored_by", "advisor"),
                ("advisor", "studies_lineage", "concept")):
        if rel not in data.edge_types:
            raise SystemExit(f"r32: relation {rel} absent from the built graph. "
                             f"Refusing to write a cell labelled as lineage.")
    print(f"[t2.2b] device={dev} {len(data.edge_types)} edge types, "
          f"{data['advisor'].num_nodes} advisor nodes "
          f"({lstats['grand_advisor_only_nodes']} grand-advisor only), "
          f"{lstats['ancestry_edges']} ancestry edges, "
          f"{lstats['lineage_concept_edges']} lineage concept edges, "
          f"{lstats['rows_with_ancestry']}/{lstats['n_rows']} rows with ancestry")

    import inspect
    if "arch" not in inspect.signature(R3.train_eval).parameters:
        raise SystemExit(
            "r32: train_eval does not accept 'arch'. Refusing to call it "
            "without the parameter, because the callee would silently build "
            "RGCN and the cell would be mislabelled.")
    EXPECT_CLASS = {"rgcn": "RGCNSym", "gat": "GATSym"}

    def fit(seed, hp):
        r = R3.train_eval(data, seed, dev, hp, budget, arch=args.arch)
        want, got = EXPECT_CLASS.get(args.arch), r.get("model_class")
        if want is None:
            raise SystemExit(f"r32: no expected model class recorded for arch "
                             f"{args.arch!r}; refusing to label the output.")
        if got != want:
            raise SystemExit(f"r32: arch {args.arch!r} requested but the model "
                             f"instantiated was {got!r}, expected {want!r}. "
                             f"Refusing to write a mislabelled cell.")
        return r

    pinned = R25.PREREG_HP if args.protocol == "prereg" else None
    if pinned is not None:
        (OUT / "winner.json").write_text(json.dumps(
            {"hp": pinned, "val_auc_pr_mean": None,
             "source": "pre-registered fixed configuration"}, indent=2))
        if args.stage == "grid":
            print(f"SKIPPED: the configuration is pinned ({pinned}), so there "
                  f"is no grid to search for cell {cell}. winner.json written.")
            return EXIT_SKIPPED
        print(f"[t2.2b] configuration pinned: {pinned}")

    grid_f = OUT / "grid.json"
    if pinned is None and (args.stage in ("grid", "all") or not grid_f.exists()):
        grid = json.loads(grid_f.read_text()) if grid_f.exists() else {}
        allhp = [dict(zip(("lr", "hidden", "layers", "dropout"), v))
                 for v in __import__("itertools").product(
                     E2.HP_GRID["lr"], E2.HP_GRID["hidden"],
                     E2.HP_GRID["layers"], E2.HP_GRID["dropout"])]
        if args.smoke:
            allhp = allhp[:1]
        for hp in allhp:
            tag = R3.hp_tag(hp)
            if tag in grid:
                continue
            vals = [fit(s, hp)["val_auc_pr"] for s in SEL_SEEDS]
            grid[tag] = {"hp": hp, "val_auc_pr_mean": float(np.mean(vals))}
            grid_f.write_text(json.dumps(grid, indent=2))
            print(f"[grid] {tag} val {np.mean(vals):.4f}", flush=True)
        best = max(grid.values(), key=lambda v: v["val_auc_pr_mean"])
        (OUT / "winner.json").write_text(json.dumps(best, indent=2))
        print(f"[t2.2b] winner {best['hp']} val {best['val_auc_pr_mean']:.4f}")

    if args.stage == "grid":
        return EXIT_OK

    best = json.loads((OUT / "winner.json").read_text())
    for s in SEEDS:
        f = OUT / f"seed{s}.json"
        if f.exists():
            print(f"[seeds] seed {s} already done, skipping")
            continue
        t = time.time()
        r = fit(s, best["hp"])
        f.write_text(json.dumps({
            "seed": s, "hp": best["hp"], "budget": budget,
            "contract": "strict_lineage", "class_weight": "train_only",
            "model_class": r["model_class"], "arch_requested": r["arch_requested"],
            "test_auc_pr": r["auc_pr"], "test_auc_roc": r["auc_roc"],
            "val_auc_pr": r["val_auc_pr"],
            "test_scores": [round(float(x), 5) for x in r["test_scores"]],
            "test_labels": [int(v) for v in r["test_labels"]],
            "seconds": round(time.time() - t, 1),
            "env": R25.env_provenance(),
        }, indent=2))
        print(f"[seeds] seed {s} test AUC-PR {r['auc_pr']:.4f} "
              f"({time.time()-t:.0f}s)", flush=True)

    if args.stage == "seeds":
        return EXIT_OK
    return aggregate(ceil, best, budget, lstats)


def aggregate(ceil, best, budget, lstats):
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

    strict = strict_summary(ARGS.arch, ARGS.protocol)
    lin_minus_strict = (round(d5p - strict["delta_vs_M5prime"], 4)
                        if strict.get("found") and
                        strict.get("delta_vs_M5prime") is not None else None)

    tbl, row = TABLE_MAP[(ARGS.arch, ARGS.protocol)]
    out = {
        "experiment": "T2_2b_lineage_contract", "field": FIELD,
        "target_table": tbl, "target_row": row,
        "model": f"{ARGS.arch}_strict_lineage",
        "cell": f"{ARGS.arch}_{ARGS.protocol}_lineage",
        "protocol": ARGS.protocol, "config_source": "lineage",
        "contract": "strict_lineage",
        "model_class": per[0].get("model_class"),
        "arch_requested": per[0].get("arch_requested"),
        "hp": best["hp"], "budget": budget,
        "changes_vs_r25": [
            "build_graph_v2 contract=strict_lineage: advisor--mentored_by-->"
            "advisor from the AFT parent map, keyed to the focal cohort",
            "advisor--studies_lineage-->concept from grand-advisor works dated "
            "at or before the focal student's t0+5"],
        "lineage_graph": lstats,
        "lineage_coverage": {
            "rows_with_ancestry_share": round(
                lstats["rows_with_ancestry"] / max(lstats["n_rows"], 1), 4)},
        "seed_mean_auc_pr": round(float(np.mean(g_ap)), 4),
        "M5_mean": round(float(np.mean(m5_ap)), 4),
        "M5prime_mean": round(float(np.mean(m5p_ap)), 4),
        "delta_vs_M5": round(d5, 4),
        "delta_vs_M5prime": round(d5p, 4),
        "wilcoxon_p_raw": round(p_raw, 6), "p_BH": round(p_bh, 6),
        "student_ci95_vs_M5prime": ci,
        "gates_fair": gates, "exceeds_fair": all(gates.values()),
        "per_seed_auc_pr": [round(v, 4) for v in g_ap],
        "matching_strict_cell": strict,
        "lineage_minus_strict_delta": lin_minus_strict,
        "env": per[0].get("env"),
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 72)
    print(f"T2.2b {FIELD} {ARGS.arch} {ARGS.protocol}, strict lineage contract")
    print(f"  lineage vs M5'   : {d5p:+.4f}  CI {ci}  p_BH {p_bh:.4f}  "
          f"exceeds={out['exceeds_fair']}")
    if strict.get("found"):
        print(f"  matching strict  : {strict['delta_vs_M5prime']:+.4f}  "
              f"CI {strict['student_ci95_vs_M5prime']}  "
              f"exceeds={strict['exceeds_fair']}")
        print(f"  lineage - strict : {lin_minus_strict:+.4f}")
    else:
        print("  matching strict  : NOT FOUND, searched")
        for p in strict["searched"]:
            print(f"      {p}")
    print("=" * 72)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
