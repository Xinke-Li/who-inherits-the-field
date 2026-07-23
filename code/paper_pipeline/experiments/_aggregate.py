"""Aggregate chunked per-seed results (e1/e9a) into the standard summary JSON + md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import stats as S


def agg_e1():
    rows = [json.loads(l) for l in
            (C.RESULTS_DIR / "e1_perseed.jsonl").read_text().splitlines()]
    per_model = {}
    for r in sorted(rows, key=lambda x: x["seed"]):
        for m, v in r["models"].items():
            per_model.setdefault(m, []).append(v)
    summary = {m: S.summarize_seeds(v) for m, v in per_model.items()}
    for m, v in per_model.items():
        if "auc_pr_ci95" in v[0]:
            summary[m]["auc_pr_ci95_seed0"] = v[0]["auc_pr_ci95"]
    out = {"experiment": "E1_baselines", "split": "temporal",
           "seeds": sorted(r["seed"] for r in rows), "summary": summary}
    (C.RESULTS_DIR / "e1_baselines.json").write_text(json.dumps(out, indent=2))
    lines = ["| Model | AUC-PR | AUC-ROC | F1 |", "|---|---|---|---|"]
    for m in ["M0_prior", "M1_logit_overlap", "M2_logit_tabular",
              "M3_gbdt_tabular", "M4_logit_tfidf", "M5_gbdt_nfa"]:
        if m not in summary:
            continue
        s = summary[m]
        f1 = f"{s['f1']['mean']:.3f}±{s['f1']['std']:.3f}" if "f1" in s else "—"
        lines.append(f"| {m} | {s['auc_pr']['mean']:.3f}±{s['auc_pr']['std']:.3f} "
                     f"| {s['auc_roc']['mean']:.3f}±{s['auc_roc']['std']:.3f} | {f1} |")
    (C.RESULTS_DIR / "e1_baselines.md").write_text("\n".join(lines))
    print("\n".join(lines))
    if "auc_pr_ci95_seed0" in summary.get("M3_gbdt_tabular", {}):
        print("M3 seed-0 AUC-PR 95% CI:", summary["M3_gbdt_tabular"]["auc_pr_ci95_seed0"])


def agg_e9a():
    p = C.RESULTS_DIR / "e9a_perseed.jsonl"
    if not p.exists():
        print("no e9a results yet"); return
    rows = [json.loads(l)["metrics"] for l in p.read_text().splitlines()]
    summary = S.summarize_seeds(rows, keys=("auc_pr", "auc_roc"))
    base = rows[0]["base_rate"]
    verdict = (abs(summary["auc_roc"]["mean"] - 0.5) < 0.03
               and abs(summary["auc_pr"]["mean"] - base) < 0.03)
    out = {"experiment": "E9a_placebo", "test_base_rate": base, "summary": summary,
           "n_seeds": len(rows),
           "verdict": "PASS - no residual leakage detected" if verdict else
                      "FAIL - placebo beats chance, investigate"}
    (C.RESULTS_DIR / "e9a_placebo.json").write_text(json.dumps(out, indent=2))
    print(f"E9a placebo: AUC-ROC {summary['auc_roc']['mean']:.3f}±{summary['auc_roc']['std']:.3f} "
          f"(target .5) | AUC-PR {summary['auc_pr']['mean']:.3f} (target {base:.3f}) -> {out['verdict']}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e1", "all"):
        agg_e1()
    if which in ("e9a", "all"):
        agg_e9a()
