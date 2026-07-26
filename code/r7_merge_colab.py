#!/usr/bin/env python3
"""R7 - merge-back verification for the Colab-run robustness jobs.

Run LOCALLY after syncing Drive's who-inherits/results/robustness/ into this
repo's results/robustness/. It:

  1. verifies completeness - all 25 theta cells and the merged theta files,
     the 32-config GPU grid (device == cuda) plus both 10-seed winner cells
     and the verdict, the 5 calibrations and 5x4 k cells and 5x2 min-score
     cells with the merged topk files, the continuous-label summary, and the
     four DONE flags;
  2. regenerates SHA256SUMS.robustness over every result JSON and every
     cache shard;
  3. prints the three decision tables: theta branch-stability, k=10
     calibration/drift + k branch-stability, and the chemistry RGCN
     symmetric verdict with the LaTeX branch it selects.

It prints tables and writes the manifest; it does NOT touch the paper - the
LaTeX branches are applied only after the tables have been reviewed.

Usage:  python code/r7_merge_colab.py [--skip-missing]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "robustness"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
THETAS = ["0.10", "0.15", "0.20", "0.25", "0.30"]
KS = [5, 10, 15, 20]
MSS = [0.2, 0.4]
problems = []


def need(path: Path, what: str):
    ok = path.exists()
    print(f"[{'OK' if ok else 'MISSING'}] {what}: {path.relative_to(ROOT)}")
    if not ok:
        problems.append(what)
    return ok


def check_complete():
    for f in FIELDS:
        for t in THETAS:
            need(R / "theta_partial" / f"{f}_theta_{t}.json", f"theta {f}@{t}")
    for t in THETAS:
        need(R / f"theta_{t}.json", f"merged theta_{t}")
    need(R / "theta_sweep_summary.json", "theta summary")
    need(R / "continuous_label_summary.json", "continuous-label summary")

    # rgcn: 32 GPU grid runs + selection + 2x10 winner + verdict
    gdir = R / "rgcn_symmetric"
    n_gpu_grid = 0
    for p in sorted(gdir.glob("rgcn_sym_grid_seed*.json")):
        if json.loads(p.read_text()).get("device") == "cuda":
            n_gpu_grid += 1
    print(f"[{'OK' if n_gpu_grid == 32 else 'MISSING'}] rgcn GPU grid: "
          f"{n_gpu_grid}/32 configs on cuda")
    if n_gpu_grid != 32:
        problems.append("rgcn GPU grid incomplete")
    need(gdir / "grid_selection.json", "rgcn grid selection")
    for prefix, tag in (("rgcn_sym_seed", "winner hgt-budget"),
                        ("rgcn_sym_orig300_seed", "winner original-budget")):
        n = sum((gdir / f"{prefix}{s}.json").exists() for s in range(10))
        bad_dev = [s for s in range(10) if (gdir / f"{prefix}{s}.json").exists()
                   and json.loads((gdir / f"{prefix}{s}.json").read_text())
                   .get("device") != "cuda"]
        print(f"[{'OK' if n == 10 and not bad_dev else 'MISSING'}] {tag}: "
              f"{n}/10 seeds{' (non-cuda: ' + str(bad_dev) + ')' if bad_dev else ''}")
        if n != 10 or bad_dev:
            problems.append(f"{tag} incomplete or off-device")
    need(R / "rgcn_symmetric_verdict.json", "rgcn verdict")

    for f in FIELDS:
        need(R / "topk_partial" / f"{f}_calibration_k10.json", f"calibration {f}")
        for k in KS:
            need(R / "topk_partial" / f"{f}_k{k}.json", f"topk {f}@k{k}")
        for ms in MSS:
            need(R / "topk_partial" / f"{f}_k10_ms{ms}.json", f"minscore {f}@{ms}")
    for k in KS:
        if k != 10:
            need(R / f"topk_{k}.json", f"merged topk_{k}")
    need(R / "topk_10.json", "merged topk_10")
    need(R / "topk_sweep_summary.json", "topk summary")

    for job in ("fetch", "theta", "rgcn", "topk"):
        need(R / f"DONE_{job}.flag", f"DONE flag {job}")


CACHE_DIRS = ("openalex_cache", "openalex_topics_cache")


def write_manifest():
    """Pin the result files, in two manifests with different provenance.

    SHA256SUMS.robustness covers exactly what ships in the git clone, so a
    reader verifies every result file the paper cites without downloading
    anything. The two OpenAlex caches are excluded on purpose: they are
    gitignored and travel through Zenodo, so listing them here would make the
    manifest unverifiable from a fresh clone. Their hashes go to
    SHA256SUMS.caches, which is small enough to keep in git and lets a reader
    verify the archive after downloading it.

    Both files are written with LF endings explicitly. On Windows the default
    text mode would emit CRLF, which makes `sha256sum -c` read every filename
    with a trailing carriage return and fail to open a single file.
    """
    repo_lines, cache_lines = [], []
    for p in sorted(R.rglob("*")):
        if not p.is_file() or p.suffix not in (".json", ".parquet", ".flag"):
            continue
        rel = p.relative_to(R).as_posix()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        target = cache_lines if rel.split("/")[0] in CACHE_DIRS else repo_lines
        target.append(f"{h}  {rel}")
    out = R / "SHA256SUMS.robustness"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(repo_lines) + "\n")
    print(f"[manifest] {out} ({len(repo_lines)} files in the clone)")
    if cache_lines:
        outc = R / "SHA256SUMS.caches"
        with open(outc, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(cache_lines) + "\n")
        print(f"[manifest] {outc} ({len(cache_lines)} Zenodo cache shards)")


def table_theta():
    s = json.loads((R / "theta_sweep_summary.json").read_text())
    print("\n== DECISION TABLE 1: theta branch stability ==")
    print(f"{'field':10} " + " ".join(f"{t:>16}" for t in THETAS) +
          "  stable(all) stable(0.15/0.25)")
    short = {"A_ADVISOR_INFORMATION_REQUIRED": "advisor-req",
             "B_SELF_PERSISTENCE_EQUIVALENT": "self-persist",
             "C_INTERMEDIATE": "advisor-adds"}
    for f in FIELDS:
        blk = s["fields"][f]
        row = " ".join(f"{short[blk['rows'][t]['branch']]:>16}" for t in THETAS)
        print(f"{f:10} {row}  {str(blk['branch_stable_all_theta']):>11} "
              f"{str(blk['branch_stable_adjacent']):>17}")
    print("\n  base rate / best-tabular AUC-PR per cell:")
    for f in FIELDS:
        blk = s["fields"][f]["rows"]
        print(f"  {f:10} " + " ".join(
            f"{blk[t]['base_rate']:.3f}/{blk[t]['best_tabular_auc_pr']:.3f}"
            for t in THETAS))


def table_topk():
    s = json.loads((R / "topk_sweep_summary.json").read_text())
    print("\n== DECISION TABLE 2a: k=10 calibration vs frozen (drift) ==")
    print(f"{'field':10} {'match<1e-6':>10} {'|d|mean':>8} {'|d|max':>8} "
          f"{'agree':>7} {'kappa':>7}")
    for f in FIELDS:
        c = s["calibration_k10"][f]
        print(f"{f:10} {c['match_within_1e6']:>10} {c['abs_delta']['mean']:>8} "
              f"{c['abs_delta']['max']:>8} {c['label_agreement']:>7} "
              f"{c['label_kappa']:>7}")
    print(f"  kappa gate: {s['kappa_gate_0.8']}")
    short = {"A_ADVISOR_INFORMATION_REQUIRED": "advisor-req",
             "B_SELF_PERSISTENCE_EQUIVALENT": "self-persist",
             "C_INTERMEDIATE": "advisor-adds"}
    print("\n== DECISION TABLE 2b: branch stability across k (rebuilt family) ==")
    cols = [f"k{k}" for k in KS] + [f"k10_ms{ms}" for ms in MSS]
    print(f"{'field':10} " + " ".join(f"{c:>14}" for c in cols) + "  stable(k)")
    for f in FIELDS:
        blk = s["fields"][f]
        row = " ".join(f"{short[blk['rows'][c]['branch']]:>14}" for c in cols)
        print(f"{f:10} {row}  {blk['branch_stable_all_k']}")


def table_rgcn():
    v = json.loads((R / "rgcn_symmetric_verdict.json").read_text())
    m = v["models"]["rgcn_symmetric"]
    print("\n== DECISION TABLE 3: chemistry RGCN symmetric verdict ==")
    print(f"  grid winner: {v['grid_selection']['winner']} "
          f"(seeds agree: {v['grid_selection']['seeds_agree']})")
    print(f"  winner @ HGT budget (200/15+wd): "
          f"{v['winner_10seed']['hgt_budget_mean_auc_pr']}")
    print(f"  winner @ original budget (300/30): "
          f"{v['winner_10seed']['original_budget_mean_auc_pr']}")
    print(f"  frozen original RGCN:              "
          f"{v['winner_10seed']['frozen_original_rgcn_mean_auc_pr']}")
    print(f"  ceilings: M5 {v['ceilings']['M5_preregistered']} | "
          f"M5' {v['ceilings']['M5_prime_val_symmetric']}")
    print(f"  delta vs M5': {m['delta_vs_M5prime']:+.4f}, "
          f"p_adj {m['p_adj_M5prime']}, student CI {m['student_ci95_vs_M5prime']}")
    print(f"  gates: {m['gates_fair']}")
    branch = ("SURVIVES - LaTeX branch 1: add 'RGCN symmetric' row to Table 4, "
              "state the crossing is robust, delete the Section 7 budget hedge"
              if v["rgcn_symmetric_exceeds_fair"] else
              "DOES NOT SURVIVE - LaTeX branch 2 (the stronger result): zero of "
              "twenty crossings survive symmetric evaluation; suite is a clean "
              "negative control; keep the asymmetric result in the appendix as "
              "a worked example of the paper's own thesis")
    print(f"  -> {branch}")


def check_fullgrid():
    """Validate the RGCN+GAT x 5 symmetric grid (notebook 05 / r17 output)."""
    g = R / "full_symmetric_grid"
    if not g.exists():
        return
    for f in FIELDS:
        need(g / f"{f}_verdict.json", f"fullgrid verdict {f}")
        for arch in ("rgcn", "gat"):
            n = sum((g / f"{f}_{arch}_sym_seed{s}.json").exists() for s in range(10))
            ok = (n == 10)
            print(f"[{'OK' if ok else 'MISSING'}] fullgrid {f} {arch}: {n}/10 seeds")
            if not ok:
                problems.append(f"fullgrid {f} {arch} incomplete")
    need(g / "DONE_fullgrid.flag", "fullgrid DONE flag")
    # every verdict must carry both architectures with the three-gate structure
    for f in FIELDS:
        p = g / f"{f}_verdict.json"
        if p.exists():
            v = json.loads(p.read_text())
            if set(v["models"]) != {"rgcn_sym", "gat_sym"}:
                problems.append(f"fullgrid {f} verdict missing an architecture")


def table_fullgrid():
    g = R / "full_symmetric_grid"
    print("\n== DECISION TABLE 4: RGCN+GAT x 5 symmetric grid (post hoc) ==")
    print(f"{'field':10} {'arch':9} {'mean':>7} {'M5p':>7} {'dM5p':>8} "
          f"{'p_adj':>7} {'student CI':>18} verdict")
    survivors = []
    for f in FIELDS:
        v = json.loads((g / f"{f}_verdict.json").read_text())
        for arch in ("rgcn_sym", "gat_sym"):
            m = v["models"][arch]
            ex = m["exceeds_fair"]
            if ex:
                survivors.append(f"{f}/{arch}")
            print(f"{f:10} {arch:9} {m['seed_mean_auc_pr']:>7} {v['M5prime_mean']:>7} "
                  f"{m['delta_vs_M5prime']:>+8.4f} {m['p_adj_M5prime']:>7.4f} "
                  f"{str(m['student_ci95_vs_M5prime']):>18} "
                  f"{'EXCEEDS' if ex else 'null'}")
    print(f"  survivors: {survivors if survivors else 'NONE'} "
          f"(chemistry/rgcn_sym is the r3 copy, expected +0.035)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-missing", action="store_true",
                    help="print whatever tables exist even if incomplete")
    args = ap.parse_args()
    check_complete()
    check_fullgrid()
    if problems and not args.skip_missing:
        sys.exit(f"\n{len(problems)} item(s) missing - sync Drive and rerun "
                 f"(or --skip-missing): {problems}")
    write_manifest()
    if (R / "theta_sweep_summary.json").exists():
        table_theta()
    if (R / "topk_sweep_summary.json").exists():
        table_topk()
    if (R / "rgcn_symmetric_verdict.json").exists():
        table_rgcn()
    if (R / "full_symmetric_grid").exists():
        table_fullgrid()
    print("\n[r7] merge-back verification finished; LaTeX edits are applied "
          "only after these tables are reviewed.")


if __name__ == "__main__":
    main()
