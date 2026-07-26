"""R16 - Topics parallel label family at two granularities (task B3).

POST-HOC ROBUSTNESS FAMILY; the pre-registered concept label remains THE
benchmark label. From the r8 topics cache, rebuild the complete label family
at subfield (about 250 labels) and topic (about 4,500 labels) granularity:
same windows, same top-10 profiles, same Jaccard construction, the same
fetch-time collapse as the concept builder (score >= 0.3, first three per
work), and a per-granularity threshold theta' calibrated on the TRAIN cohort
so the family's base rate matches the concept label's per discipline
(deterministic atom threshold, no randomization: labels stay clean).

Per (discipline, granularity), r1_theta_sweep.run_theta is reused verbatim
on a dataframe whose profile-derived columns are rebuilt from topics
(early_concepts and adv_profile carry granularity ids as tokens, so the
TF-IDF arms work unchanged): the tabular ladder M0..M5 at 10 seeds, the
four certificates, and the mechanism branch under the unchanged 0.05
thresholds. Label agreement with the concept label is reported raw
(theta 0.2 on the topics overlap) and calibrated (theta'), with Cohen's
kappa, against the frozen concept label (whose rebuilt-family drift is
bounded at kappa >= 0.995 in Appendix G).

Usage: DATASET=<f> DATASET_PATH=... python r16_topics_parallel.py
       python r16_topics_parallel.py --merge
Output: results/robustness/topics_partial/<field>_<gran>.json,
        results/robustness/topics_parallel.json (--merge)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D

from r1_theta_sweep import run_theta
from r6_topk_sweep import profile_counts, topk_set
from e14_self_persistence import (LEGACY_STUDENT_TFIDF, jaccard,
                                  student_tfidf)
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO / "results" / "robustness" / "openalex_topics_cache"
OUT_DIR = REPO / "results" / "robustness"
PARTIAL = OUT_DIR / "topics_partial"
FIELDS = ["econ", "math", "physics", "neuro", "chemistry"]
GRANULARITIES = {"topic": 0, "subfield": 1}   # index into [t_id, sf_id, score]
MIN_SCORE, TOP_PER_WORK, K = 0.3, 3, 10


def load_topics_for(aids):
    out = {}
    for p in sorted(CACHE_DIR.glob("topics_*.parquet")):
        df = pd.read_parquet(p)
        hit = df[df.aid.isin(aids)]
        for aid, wj in zip(hit.aid, hit.works_json):
            out[aid] = json.loads(wj)
    return out


def gran_view(works, gidx):
    """Builder-analog collapse at one granularity: score >= 0.3, first three,
    in API order (descending score)."""
    return [{"year": w["year"],
             "concepts": [t[gidx] for t in w["topics"]
                          if t[gidx] is not None and t[2] >= MIN_SCORE][:TOP_PER_WORK]}
            for w in works]


def calibrate_atom(lo_train, target):
    """Deterministic atom threshold minimizing |train rate - target|."""
    atoms = np.unique(np.concatenate([[0.0], lo_train]))
    rates = np.array([(lo_train > a).mean() for a in atoms])
    j = int(np.argmin(np.abs(rates - target)))
    return float(atoms[j]), float(rates[j])


def main_field():
    field = C.CLEAN_DATASET.stem.replace("clean_dataset_", "")
    PARTIAL.mkdir(parents=True, exist_ok=True)
    df = D.load_dataset()
    ds = D.temporal_split(df)
    is_train = (ds.split == "train").values
    y_concept = df.y.values
    base_rate = float(y_concept.mean())

    aids = set(df.st_openalex_id) | set(df.adv_openalex_id)
    print(f"[r16] {field}: loading topics cache for {len(aids)} authors", flush=True)
    cache = load_topics_for(aids)
    print(f"[r16] {field}: {len(cache)} cached", flush=True)

    for gran, gidx in GRANULARITIES.items():
        outp = PARTIAL / f"{field}_{gran}.json"
        if outp.exists():
            print(f"[r16] {field}/{gran} done, skipping", flush=True)
            continue
        view = {a: gran_view(cache[a], gidx) for a in aids if a in cache}
        rows, keep = [], []
        for r in df.itertuples():
            sw, aw = view.get(r.st_openalex_id), view.get(r.adv_openalex_id)
            if not sw or not aw:
                keep.append(False); rows.append(None); continue
            cut = r.t0 + C.EARLY_YEARS
            e_cnt, n_e = profile_counts(sw, r.t0, cut)
            l_cnt, n_l = profile_counts(sw, cut + 1, r.t0 + C.LATE_YEARS)
            a_cnt, n_a = profile_counts(aw, None, cut)
            early, late, advp = topk_set(e_cnt, K), topk_set(l_cnt, K), topk_set(a_cnt, K)
            if not early:
                keep.append(False); rows.append(None); continue
            keep.append(True)
            rows.append({"early": sorted(early), "adv": sorted(advp),
                         "eo": jaccard(early, advp), "lo": jaccard(late, advp),
                         "n_e": n_e, "br_e": len(e_cnt), "n_a": n_a,
                         "br_a": len(a_cnt)})
        keep = np.array(keep)
        idx = np.where(keep)[0]
        lo = np.array([rows[i]["lo"] for i in idx])
        theta_g, tr_rate = calibrate_atom(lo[is_train[idx]], base_rate)

        dfg = df.iloc[idx].reset_index(drop=True).copy()
        dfg["early_concepts"] = [rows[i]["early"] for i in idx]
        dfg["adv_profile"] = [rows[i]["adv"] for i in idx]
        dfg["early_overlap"] = [round(rows[i]["eo"], 4) for i in idx]
        dfg["late_overlap"] = lo
        dfg["early_prod"] = [rows[i]["n_e"] for i in idx]
        dfg["early_breadth"] = [rows[i]["br_e"] for i in idx]
        dfg["adv_early_prod"] = [rows[i]["n_a"] for i in idx]
        dfg["adv_early_breadth"] = [rows[i]["br_a"] for i in idx]

        # agreement with the concept label
        yc = y_concept[idx]
        y_raw = (lo > C.JACCARD_THETA).astype(int)
        y_cal = (lo > theta_g).astype(int)
        agreement = {
            "n": int(len(idx)), "coverage": round(float(keep.mean()), 4),
            "theta_calibrated": round(theta_g, 4),
            "train_rate_achieved": round(tr_rate, 4),
            "raw_theta02": {"agreement": round(float((yc == y_raw).mean()), 4),
                            "kappa": round(float(cohen_kappa_score(yc, y_raw)), 4),
                            "base_rate": round(float(y_raw.mean()), 4)},
            "calibrated": {"agreement": round(float((yc == y_cal).mean()), 4),
                           "kappa": round(float(cohen_kappa_score(yc, y_cal)), 4),
                           "base_rate": round(float(y_cal.mean()), 4)},
        }
        print(f"[r16] {field}/{gran}: theta'={theta_g:.3f} kappa_cal="
              f"{agreement['calibrated']['kappa']}", flush=True)

        X_st = student_tfidf(D.temporal_split(dfg.copy()),
                             legacy=LEGACY_STUDENT_TFIDF)
        res = run_theta(dfg, theta_g, X_st)
        res.update({"field": field, "granularity": gran,
                    "provenance": "topics family (r8 cache); read within-family",
                    "concept_label_agreement": agreement})
        outp.write_text(json.dumps(res, indent=2))
        print(f"[r16] {field}/{gran}: branch {res['mechanism_branch']} best "
              f"{res['best_pure_tabular']['auc_pr']:.4f}", flush=True)


def main_merge():
    out = {"experiment": "R16_topics_parallel",
           "note": ("parallel robustness family; the pre-registered concept "
                    "label remains the benchmark label; within-family reading "
                    "(r6 convention)"),
           "fields": {}}
    for f in FIELDS:
        out["fields"][f] = {}
        for g in GRANULARITIES:
            p = PARTIAL / f"{f}_{g}.json"
            if p.exists():
                d = json.loads(p.read_text())
                out["fields"][f][g] = {
                    "agreement": d["concept_label_agreement"],
                    "base_rate": d["base_rate"],
                    "best_tabular_auc_pr": round(d["best_pure_tabular"]["auc_pr"], 4),
                    "ladder": {m: round(d["ladder"][m]["auc_pr"]["mean"], 4)
                               for m in d["ladder"]},
                    "scramble": d["y_scrambling"]["verdict"],
                    "disjoint": d["advisor_disjoint"]["verdict"],
                    "placebo_gap": d["advisor_placebo"]["gap_true_minus_cohort"],
                    "floor": d["self_persistence"]["verdict"]["best_student_auc_pr"],
                    "branch": d["mechanism_branch"]}
    (OUT_DIR / "topics_parallel.json").write_text(json.dumps(out, indent=2))
    for f in FIELDS:
        for g, v in out["fields"][f].items():
            print(f"{f:10} {g:9} kappa {v['agreement']['calibrated']['kappa']:.3f} "
                  f"best {v['best_tabular_auc_pr']:.3f} branch {v['branch']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        main_merge()
    else:
        main_field()
