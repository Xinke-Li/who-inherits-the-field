#!/usr/bin/env python3
"""T2.2a: the tabular genealogy arm, M6, under three gates.

Reviewer B1: the multi-generation genealogy of Figure 2 enters no model, so the
graph comparison is a weaker test than the paper claims. This adds it on the
tabular side and then tries hard to kill the result.

GATE 1  harness validation. Reproduce the frozen M5 exactly (same split, seeds,
        NFA construction, hyperparameters) and print the delta against the
        published Table 3 values. Nothing downstream is meaningful unless this
        matches. The first version of this script used bare
        HistGradientBoostingClassifier defaults (max_iter=100, early_stopping
        'auto') instead of the frozen max_iter=500, early_stopping=True,
        validation_fraction=0.15, which undertrained the baseline and
        manufactured a lift.

GATE 2  eq. (2), not seed counts. Paired Wilcoxon per seed, Benjamini-Hochberg
        across the per-discipline family, and a paired student-level bootstrap
        over test students with 2000 draws. All three gates must pass.

GATE 3  missingness control. The lift appeared where grand-advisor coverage was
        lowest, which is the signature of a missingness indicator rather than of
        genealogy information. An arm of M5 plus one binary "has a grand-advisor
        with cached works" feature isolates that channel.

Output: results/revision/T2_2a_genealogy_tabular/

  python code/r23_genealogy_tabular.py
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "T2_2a_genealogy_tabular"
CACHE = ROOT / "results" / "robustness" / "openalex_cache"
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
SEEDS = list(range(10))
EARLY, TOPK, N_BOOT = 5, 10, 2000

# paper Table 3, M5 row, AUC-PR
FROZEN_M5 = {"econ": 0.339, "math": 0.398, "neuro": 0.423,
             "physics": 0.644, "chemistry": 0.531}

GFEATS = ["grand_adv_early_prod", "grand_adv_early_breadth",
          "grand_adv_career_age_at_t0", "jac_student_grandadv",
          "jac_advisor_grandadv", "lineage_depth", "advisor_fanout_at_t0"]
MISSFEAT = ["has_grandadv_works"]


def jac(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A or B) else 0.0


def fast_auc_pr(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def load_cache(aids):
    want, got = set(a for a in aids if isinstance(a, str)), {}
    for f in sorted(glob.glob(str(CACHE / "*.parquet"))):
        df = pd.read_parquet(f)
        hit = df[df.aid.isin(want)]
        for aid, wj in zip(hit.aid, hit.works_json):
            try:
                works = json.loads(wj)
            except Exception:
                continue
            got[aid] = [(w.get("year"), [c[0] for c in (w.get("concepts") or [])])
                        for w in works if w.get("year")]
        if len(got) == len(want):
            break
    return got


def profile_asof(works, cutoff):
    sel = [(y, cs) for (y, cs) in works if y is not None and y <= cutoff]
    assert all(y <= cutoff for y, _ in sel), "leakage guard: work after cutoff"
    if not sel:
        return 0, 0, [], None
    counts = {}
    for _, cs in sel:
        for c in cs:
            counts[c] = counts.get(c, 0) + 1
    top = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOPK]]
    return len(sel), len(counts), top, min(y for y, _ in sel)


def build_genealogy(field, dsp):
    pr = pd.read_parquet(ROOT / "data" / f"pairs_resolved_{field}.parquet")

    def bare(x):
        return x.rsplit("/", 1)[-1] if isinstance(x, str) else x

    par, par_aid = {}, {}
    for sp, ap, aoa in zip(pr.student_pid, pr.advisor_pid, pr.adv_openalex_id):
        if pd.notna(sp) and pd.notna(ap) and sp not in par:
            par[sp], par_aid[sp] = ap, bare(aoa)

    g = pd.DataFrame({"student_pid": dsp.student_pid})
    g["grand_adv_aid"] = dsp.advisor_pid.map(par_aid).values

    def depth(pid, cap=12):
        d, seen = 0, set()
        while pid in par and pid not in seen and d < cap:
            seen.add(pid); pid = par[pid]; d += 1
        return d
    g["lineage_depth"] = [depth(p) for p in dsp.student_pid]

    t0_of = dict(zip(dsp.student_pid, dsp.t0))
    by_adv = {}
    for sp, ap in zip(dsp.student_pid, dsp.advisor_pid):
        by_adv.setdefault(ap, []).append(t0_of[sp])
    g["advisor_fanout_at_t0"] = [
        sum(1 for t in by_adv.get(ap, []) if t < t0)
        for ap, t0 in zip(dsp.advisor_pid, dsp.t0)]

    cache = load_cache(g.grand_adv_aid.dropna().unique())
    prod, breadth, age, js, ja, has = [], [], [], [], [], []
    for aid, t0, sc, apf in zip(g.grand_adv_aid, dsp.t0,
                                dsp.early_concepts, dsp.adv_profile):
        works = cache.get(aid) if isinstance(aid, str) else None
        if not works:
            prod.append(0); breadth.append(0); age.append(0)
            js.append(0.0); ja.append(0.0); has.append(0)
            continue
        p, b, top, first = profile_asof(works, int(t0) + EARLY)
        prod.append(p); breadth.append(b)
        age.append(max(0, int(t0) - first) if first else 0)
        js.append(jac(list(sc), top)); ja.append(jac(list(apf), top))
        has.append(1 if p > 0 else 0)
    g["grand_adv_early_prod"] = prod
    g["grand_adv_early_breadth"] = breadth
    g["grand_adv_career_age_at_t0"] = age
    g["jac_student_grandadv"] = js
    g["jac_advisor_grandadv"] = ja
    g["has_grandadv_works"] = has
    return g, float(g.grand_adv_aid.notna().mean()), float(np.mean(has))


def run_one(field):
    sys.path.insert(0, str(ROOT / "code" / "paper_pipeline"))
    from scipy.stats import wilcoxon
    from sklearn.ensemble import HistGradientBoostingClassifier
    from utils import data as D
    from utils import stats as S

    dsp = D.temporal_split(D.load_dataset())
    g, cov, withworks = build_genealogy(field, dsp)
    for c in GFEATS + MISSFEAT:
        dsp[c] = g[c].values

    Xt, _ = D.build_features(dsp, concepts="none")
    nfa = D.build_nfa_features(dsp)
    base = np.hstack([Xt, nfa.values.astype(float)])
    arms = {
        "M5_gbdt_nfa": base,
        "M6_gbdt_nfa_genealogy": np.hstack([base, dsp[GFEATS].values.astype(float)]),
        "M5_plus_missingness": np.hstack([base, dsp[MISSFEAT].values.astype(float)]),
    }

    scores, ap = {}, {}
    for name, X in arms.items():
        sp = D.split_xy(dsp, X)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = sp["train"], sp["val"], sp["test"]
        scores[name], ap[name] = [], []
        for seed in SEEDS:
            # frozen hyperparameters, verbatim from e1_baselines.py:90-91
            m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                               early_stopping=True,
                                               validation_fraction=0.15)
            m.fit(Xtr, ytr)
            s = m.predict_proba(Xte)[:, 1]
            scores[name].append(s)
            ap[name].append(fast_auc_pr(yte, s))

    # ---------------- GATE 1 ----------------
    m5_mean = float(np.mean(ap["M5_gbdt_nfa"]))
    gate1 = {"reproduced_M5_auc_pr": round(m5_mean, 4),
             "frozen_M5_auc_pr": FROZEN_M5[field],
             "delta": round(m5_mean - FROZEN_M5[field], 4),
             "matches_3dp": abs(m5_mean - FROZEN_M5[field]) < 5e-4}

    # ---------------- GATE 2 ----------------
    rng = np.random.default_rng(0)
    n = len(yte)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = {name: np.empty((len(SEEDS), N_BOOT)) for name in arms}
    for b in range(N_BOOT):
        i = idx[b]; yb = yte[i]
        if yb.sum() == 0 or yb.sum() == len(yb):
            for name in arms:
                boot[name][:, b] = np.nan
            continue
        for name in arms:
            for s in range(len(SEEDS)):
                boot[name][s, b] = fast_auc_pr(yb, scores[name][s][i])

    fam = ["M6_gbdt_nfa_genealogy", "M5_plus_missingness"]
    praw = []
    for name in fam:
        d = np.array(ap[name]) - np.array(ap["M5_gbdt_nfa"])
        praw.append(float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
    # bh_correction returns (adjusted p-values, reject flags); element 0 is
    # the adjusted p-values. Taking element 1 yields booleans and inverts the gate.
    padj = list(S.bh_correction(praw)[0])

    verdicts = {}
    for k, name in enumerate(fam):
        pooled = np.nanmean(boot[name] - boot["M5_gbdt_nfa"], axis=0)
        ci = [round(float(np.nanpercentile(pooled, 2.5)), 4),
              round(float(np.nanpercentile(pooled, 97.5)), 4)]
        dbar = float(np.mean(ap[name]) - m5_mean)
        gates = {"mean_gt_ceiling": bool(dbar > 0),
                 "p_adj_lt_0.05": bool(padj[k] < 0.05),
                 "student_ci_lower_gt_0": bool(ci[0] > 0)}
        verdicts[name] = {
            "mean_auc_pr": round(float(np.mean(ap[name])), 4),
            "delta_vs_M5": round(dbar, 4),
            "wilcoxon_p_raw": round(praw[k], 6),
            "p_BH": round(float(padj[k]), 6),
            "student_ci95": ci,
            "seeds_improved": int(sum(1 for x, y in
                                      zip(ap[name], ap["M5_gbdt_nfa"]) if x > y)),
            "gates": gates,
            "exceeds": all(gates.values()),
        }

    out = {"field": field, "n_test": int(n),
           "grand_advisor_coverage": round(cov, 4),
           "grand_advisor_with_cached_works": round(withworks, 4),
           "features_added": GFEATS,
           "gate1_harness_validation": gate1,
           "gate2_eq2_verdicts": verdicts,
           "gate3_note": ("M5_plus_missingness isolates the single binary "
                          "has-a-grand-advisor-with-works channel"),
           "M5_per_seed_auc_pr": [round(v, 4) for v in ap["M5_gbdt_nfa"]]}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{field}.json", "w"), indent=2)

    v6, vm = verdicts["M6_gbdt_nfa_genealogy"], verdicts["M5_plus_missingness"]
    print(f"[G1] {field:10} M5 {m5_mean:.4f} vs frozen {FROZEN_M5[field]:.3f} "
          f"delta {gate1['delta']:+.4f} match3dp={gate1['matches_3dp']}")
    print(f"[G2] {field:10} M6 d{v6['delta_vs_M5']:+.4f} pBH {v6['p_BH']:.4f} "
          f"CI {v6['student_ci95']} exceeds={v6['exceeds']}")
    print(f"[G3] {field:10} miss d{vm['delta_vs_M5']:+.4f} CI {vm['student_ci95']} "
          f"exceeds={vm['exceeds']}")
    return 0


def main():
    if os.environ.get("R23_FIELD"):
        return run_one(os.environ["R23_FIELD"])
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        env = dict(os.environ)
        env.update(R23_FIELD=f, DATASET=f,
                   DATASET_PATH=str(ROOT / "data" / f"clean_dataset_{f}.parquet"))
        env.pop("NEURO_DATASET", None)
        subprocess.call([sys.executable, str(Path(__file__).resolve())], env=env)

    per = {f: json.load(open(OUT / f"{f}.json")) for f in FIELDS
           if (OUT / f"{f}.json").exists()}
    summary = {
        "task": "T2.2a", "reviewer_point": "B1",
        "gate1_all_match_3dp": all(r["gate1_harness_validation"]["matches_3dp"]
                                   for r in per.values()),
        "gate1": {f: r["gate1_harness_validation"] for f, r in per.items()},
        "gate2": {f: r["gate2_eq2_verdicts"]["M6_gbdt_nfa_genealogy"]
                  for f, r in per.items()},
        "gate3": {f: r["gate2_eq2_verdicts"]["M5_plus_missingness"]
                  for f, r in per.items()},
        "ceiling_moves": [f for f, r in per.items()
                          if r["gate2_eq2_verdicts"]["M6_gbdt_nfa_genealogy"]["exceeds"]],
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n[T2.2a] gate 1 all match to 3dp: {summary['gate1_all_match_3dp']}")
    print(f"[T2.2a] disciplines where M6 passes all three eq.(2) gates: "
          f"{summary['ceiling_moves'] or 'none'}")
    print(f"[T2.2a] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
