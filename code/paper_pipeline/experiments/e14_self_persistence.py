"""E14 - Self-persistence / student-only control (W1 response, added 2026-07-14).

Reviewer attack this experiment answers: E10 shows the predictive signal is
advisor-SPECIFIC (true-advisor overlap beats placebo-advisor overlap), but it
does not rule out a subtler tautology: "the student simply stays where they
started, and that place happens to be the advisor's area." E10 varies WHICH
advisor profile the feature uses; E14 removes the advisor from the feature set
altogether. Together the two controls bracket what "inheritance" can mean:
E10 answers "is the signal advisor-specific?", E14 answers "is any advisor
information needed at all, or does the student's own early position suffice?"

TEMPORAL CONTRACT NOTE. The reviewer-suggested feature J(C_early, C_late) uses
the late window and is therefore banned as a model feature. Everything in part
(a) is computed from pre-window information only:
  early_typicality        J(C_early, top-10 concepts by document frequency over
                          ALL students' early profiles)  - the student-side
                          mirror of E10's deterministic field_mean feature
  early_typicality_cohort same, but the top-10 is computed over the cohort pool
                          |t0_j - t0_i| <= 3 (mirrors E10's cohort matching)
  early_concentration     Herfindahl index of the student's early-window concept
                          occurrence counts (from the per-author works store)
Both typicality profiles are fit on all rows, following the pipeline's TF-IDF
convention (vocabulary from pre-window text only); to the extent this is
generous, it only inflates the student-only models, which is conservative for
the branch-A reading below.

Part (a) - student-only prediction ladder (ZERO advisor information):
  M1s_typicality  logistic on early_typicality alone     (M1/L1 analogue)
  M1s_scalars     logistic on the 5 student scalars
  M4s_tfidf       logistic on student scalars + student-side TF-IDF only
                  (build_features' vec_st branch with vec_ad and all adv_*,
                  coauth columns stripped)                (M4 analogue)
  M3s_gbdt        GBDT on all student-side features       (M3 analogue)
Same temporal split, thresholds tuned on VAL only, 10 seeds, AUC-PR primary.
References recomputed in-script: M0 prior and M1 (logistic on early_overlap);
M1 must reproduce the frozen E1 value to 1e-3 (deterministic).

PRE-REGISTERED DECISION RULE (written before running):
  Let best = max seed-mean AUC-PR over the four student-only models,
      M1   = the recomputed deterministic M1 AUC-PR,
      band = max placebo L1 AUC-PR from the frozen E10 json (econ 0.182),
      CI   = paired test-set bootstrap 95% CI of (M1 - best), 2000 draws, seed 0.
  Branch A "ADVISOR_INFORMATION_REQUIRED":
      best <= band + 0.05  AND  CI excludes 0 (lower bound > 0).
      -> the predictable signal requires knowing WHO the advisor is; the
         first finding strengthens (paper keeps the inheritance framing,
         bounded by the E10+E14 certificate pair).
  Branch B "SELF_PERSISTENCE_EQUIVALENT":
      CI includes 0  OR  best >= M1 - 0.05.
      -> student-only models recover the advisor-informed signal; the paper
         must systematically reframe "inheritance" as "early topical
         positioning" (P0-4 branch 2).
  Branch C "INTERMEDIATE" (neither A nor B; A checked first):
      -> report the partial student-side signal honestly: some of the
         predictable variance is self-persistence, the remainder requires
         advisor identity; narrative keeps both certificates and says so.

Part (b) - label decomposition (descriptive; late info allowed HERE because it
analyses the LABEL, it never enters a feature):
  C_late is recomputed per student from the per-author works store with the
  builder's exact profile() logic (top-10 by Counter.most_common over works in
  store order, late window [t0+6, t0+15]). Self-check: the recomputed
  round(J(C_late, A), 4) must equal the frozen late_overlap for EVERY row
  (tolerance 1e-9) - asserted, otherwise the decomposition is not talking
  about the paper's label.
  Reported: (1) J(C_late, C_early) distribution by y; (2) among y=1, the share
  of C_late ∩ A concepts already present in C_early; (3) alternative label
  y_self = 1[J(C_late, C_early) > 0.2]: confusion matrix vs y, Cohen's kappa,
  and M2/M3 (full tabular) AUCs predicting y_self vs predicting y.

Part (c) - E9e split into sides (answers "E9e never separated the sides"):
  GBDT (M3 analogue), 10 seeds, three drop-combos of the tabular set:
    no_advisor      drop early_overlap + all adv_*  (student + coauth kept)
    advisor_only    keep only adv_early_prod, adv_early_breadth,
                    adv_career_age_at_t0 (early_overlap dropped too: it mixes
                    the student side into the pair)
    no_coauth       drop coauth_early, coauth_early_n

Output: results_econ/e14_self_persistence.json (or results_neuro/ under the
NEURO_DATASET override). Runs on the frozen table + works store, read-only.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D
from utils import stats as S

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COHORT_WINDOW = 3          # mirrors E10
SELF_THETA = C.JACCARD_THETA   # y_self threshold = the paper's theta (0.2)


# ---------- builder-exact primitives (must not drift from the builder) ----------

def profile_counts(works, y0=None, y1=None):
    """Concept occurrence counts in a year window - the builder's counting loop."""
    cnt = Counter()
    n = 0
    for w in works:
        y = w.get("year")
        if y is None:
            continue
        if y0 is not None and y < y0:
            continue
        if y1 is not None and y > y1:
            continue
        n += 1
        for c in w["concepts"]:
            cnt[c] += 1
    return cnt, n


def topk_set(cnt, topk=10):
    return set(c for c, _ in cnt.most_common(topk))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def load_student_windows(df):
    """One streaming pass over the works store: per-student early-window concept
    counts (for the Herfindahl feature) and the recomputed late top-10 profile.
    Works are consumed in store order, reproducing the builder's Counter
    insertion order (ties in most_common break identically)."""
    need = {}
    for r in df.itertuples():
        need[r.st_openalex_id] = r.t0
    early_cnt, late_set, late_n = {}, {}, {}
    n_lines = 0
    with open(C.WORKS_STORE) as f:
        for line in f:
            n_lines += 1
            # cheap prefix parse: {"aid": "A....", ...} - aid starts at char 9
            try:
                q = line.index('"', 9)
                aid = line[9:q]
            except ValueError:
                continue
            if aid not in need:
                continue
            works = json.loads(line)["works"]
            t0 = need[aid]
            ec, _ = profile_counts(works, t0, t0 + C.EARLY_YEARS)
            lc, nl = profile_counts(works, t0 + C.EARLY_YEARS + 1, t0 + C.LATE_YEARS)
            early_cnt[aid] = ec
            late_set[aid] = topk_set(lc)
            late_n[aid] = nl
    missing = set(need) - set(late_set)
    assert not missing, f"{len(missing)} students missing from works store"
    print(f"[e14] works store: {n_lines} authors scanned, {len(late_set)} students matched")
    return early_cnt, late_set


# ---------- part (a): student-only features ----------

def herfindahl(cnt):
    tot = sum(cnt.values())
    if tot == 0:
        return 1.0
    return float(sum((v / tot) ** 2 for v in cnt.values()))


def student_features(df, early_cnt):
    """The three new pre-window student-side scalars (+ the existing two)."""
    early_sets = [set(l) for l in df.early_concepts]

    # discipline-level typicality profile: document frequency over student
    # early profiles (each concept once per student), top-10
    disc = Counter(c for s in early_sets for c in s)
    disc_top = topk_set(disc)
    typ = np.array([jaccard(s, disc_top) for s in early_sets])

    # cohort-matched version: top-10 within the |t0 - t0_i| <= 3 pool
    by_year = {}
    for s, t in zip(early_sets, df.t0.values):
        by_year.setdefault(int(t), Counter()).update(s)
    cohort_top = {}
    for t in sorted(by_year):
        pool = Counter()
        for u in range(t - COHORT_WINDOW, t + COHORT_WINDOW + 1):
            if u in by_year:
                pool.update(by_year[u])
        cohort_top[t] = topk_set(pool)
    typ_cohort = np.array([jaccard(s, cohort_top[int(t)])
                           for s, t in zip(early_sets, df.t0.values)])

    conc = np.array([herfindahl(early_cnt[a]) for a in df.st_openalex_id])

    F = pd.DataFrame({
        "early_prod": df.early_prod.astype(float).values,
        "early_breadth": df.early_breadth.astype(float).values,
        "early_typicality": typ,
        "early_typicality_cohort": typ_cohort,
        "early_concentration": conc,
    }, index=df.index)
    return F


def student_tfidf(df, max_features=2000):
    """build_features' vec_st branch only - no advisor vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = df.early_concepts.apply(lambda l: " ".join(c.replace(" ", "_") for c in l))
    vec = TfidfVectorizer(max_features=max_features, token_pattern=r"\S+")
    return vec.fit_transform(docs)


def fit_logit(Xtr, ytr, Xva, yva, Xte, yte, seed, sparse=False, C_reg=1.0):
    scaler = StandardScaler(with_mean=not sparse)
    kw = dict(max_iter=1000, random_state=seed)
    if sparse:
        kw.update(C=C_reg, solver="liblinear")
    m = make_pipeline(scaler, LogisticRegression(**kw))
    m.fit(Xtr, ytr)
    thr = S.best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
    scores = m.predict_proba(Xte)[:, 1]
    return S.evaluate(yte, scores, thr), scores


def fit_gbdt(Xtr, ytr, Xva, yva, Xte, yte, seed):
    m = HistGradientBoostingClassifier(random_state=seed, max_iter=500,
                                       early_stopping=True, validation_fraction=0.15)
    m.fit(Xtr, ytr)
    thr = S.best_f1_threshold(yva, m.predict_proba(Xva)[:, 1])
    scores = m.predict_proba(Xte)[:, 1]
    return S.evaluate(yte, scores, thr), scores


def paired_bootstrap_delta(y, s_ref, s_alt, n_boot=2000, seed=0):
    """Percentile bootstrap CI over test rows of AP(ref) - AP(alt), paired draws."""
    rng = np.random.default_rng(seed)
    y, s_ref, s_alt = map(np.asarray, (y, s_ref, s_alt))
    n, vals = len(y), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(average_precision_score(y[idx], s_ref[idx])
                    - average_precision_score(y[idx], s_alt[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"delta_mean": float(np.mean(vals)), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run_ladder(df_split, F_student, X_st_tfidf):
    """Part (a): 10-seed student-only ladder + deterministic references."""
    from scipy import sparse as sp
    masks = {k: (df_split.split == k).values for k in ("train", "val", "test")}
    y = {k: df_split.loc[m, "y"].values for k, m in masks.items()}

    # references
    Xt, cols = D.build_features(df_split, concepts="none")
    i_ov = cols.index("early_overlap")
    Xov = Xt[:, [i_ov]]

    designs = {
        "M1s_typicality": ("logit", F_student[["early_typicality"]].values, {}),
        "M1s_scalars": ("logit", F_student.values, {}),
        "M4s_tfidf": ("logit_sparse",
                      sp.hstack([sp.csr_matrix(F_student[["early_prod", "early_breadth"]].values),
                                 X_st_tfidf]).tocsr(), {"C_reg": 0.5}),
        "M3s_gbdt": ("gbdt",
                     np.hstack([F_student.values,
                                X_st_tfidf.toarray()]).astype(np.float32), {}),
    }

    per_model, test_scores = {}, {}
    # deterministic M1 reference (also gives per-row scores for the bootstrap)
    r, s_m1 = fit_logit(Xov[masks["train"]], y["train"], Xov[masks["val"]], y["val"],
                        Xov[masks["test"]], y["test"], seed=0)
    per_model["M1_ref_logit_overlap"] = [r]
    test_scores["M1_ref_logit_overlap"] = s_m1
    prior = np.full(masks["test"].sum(), y["train"].mean())
    per_model["M0_prior"] = [S.evaluate(y["test"], prior)]

    for name, (kind, X, kw) in designs.items():
        runs = []
        for seed in C.SEEDS:
            a = (X[masks["train"]], y["train"], X[masks["val"]], y["val"],
                 X[masks["test"]], y["test"])
            if kind == "logit":
                r, s = fit_logit(*a, seed=seed)
            elif kind == "logit_sparse":
                r, s = fit_logit(*a, seed=seed, sparse=True, **kw)
            else:
                r, s = fit_gbdt(*a, seed=seed)
            runs.append(r)
            if seed == C.SEEDS[0]:
                test_scores[name] = s
        per_model[name] = runs
        print(f"[e14a] {name}: AUC-PR {np.mean([r['auc_pr'] for r in runs]):.4f}")

    summary = {m: S.summarize_seeds(v) for m, v in per_model.items()}

    # frozen-value cross-checks
    frozen = {}
    e1p = C.RESULTS_DIR / "e1_baselines.json"
    if e1p.exists():
        e1 = json.loads(e1p.read_text())
        frozen["M1_frozen_auc_pr"] = e1["summary"]["M1_logit_overlap"]["auc_pr"]["mean"]
        dev = abs(summary["M1_ref_logit_overlap"]["auc_pr"]["mean"] - frozen["M1_frozen_auc_pr"])
        assert dev < 1e-3, f"M1 reference does not reproduce frozen E1 value (dev {dev})"
    e10p = C.RESULTS_DIR / "e10_advisor_placebo.json"
    band = None
    if e10p.exists():
        e10 = json.loads(e10p.read_text())
        band = max(e10["summary"][v]["L1_logit_overlap"]["auc_pr"]["mean"]
                   for v in ("placebo_cohort", "placebo_random", "field_mean"))
        frozen["E10_placebo_L1_band_max"] = band

    # pre-registered comparisons vs M1
    m1_pr = summary["M1_ref_logit_overlap"]["auc_pr"]["mean"]
    comparisons = {}
    for name in designs:
        comparisons[name] = {
            "vs_M1_bootstrap": paired_bootstrap_delta(y["test"], s_m1, test_scores[name]),
            "seed_mean_auc_pr": summary[name]["auc_pr"]["mean"],
        }
        seed_prs = [r["auc_pr"] for r in per_model[name]]
        if np.std(seed_prs) > 0:  # stochastic model: one-sample signed-rank vs deterministic M1
            from scipy import stats as sps
            w = sps.wilcoxon(np.array(seed_prs) - m1_pr)
            comparisons[name]["one_sample_wilcoxon_vs_M1"] = {
                "p": float(w.pvalue), "note": "10-seed one-sample signed-rank against the deterministic M1 value"}

    # verdict
    best_name = max(designs, key=lambda n: summary[n]["auc_pr"]["mean"])
    best_pr = summary[best_name]["auc_pr"]["mean"]
    ci = comparisons[best_name]["vs_M1_bootstrap"]
    band_hi = (band if band is not None else 0.19) + 0.05
    if best_pr <= band_hi and ci["ci95"][0] > 0:
        branch = "A_ADVISOR_INFORMATION_REQUIRED"
    elif (ci["ci95"][0] <= 0 <= ci["ci95"][1]) or best_pr >= m1_pr - 0.05:
        branch = "B_SELF_PERSISTENCE_EQUIVALENT"
    else:
        branch = "C_INTERMEDIATE"

    return {"summary": summary, "per_seed": per_model, "comparisons": comparisons,
            "frozen_refs": frozen,
            "verdict": {"branch": branch, "best_student_model": best_name,
                        "best_student_auc_pr": round(best_pr, 4),
                        "M1_auc_pr": round(m1_pr, 4),
                        "placebo_band_plus_margin": round(band_hi, 4)}}


# ---------- part (b): label decomposition ----------

def run_decomposition(df_split, late_set):
    early_sets = [set(l) for l in df_split.early_concepts]
    adv_sets = [set(l) for l in df_split.adv_profile]
    lates = [late_set[a] for a in df_split.st_openalex_id]

    # byte-level self-check against the frozen label column
    rec = np.array([round(jaccard(l, a), 4) for l, a in zip(lates, adv_sets)])
    dev = np.abs(rec - df_split.late_overlap.values)
    n_bad = int((dev > 1e-9).sum())
    assert n_bad == 0, (f"late_overlap NOT reproduced for {n_bad} rows "
                        f"(max dev {dev.max()}); decomposition would be about a different label")
    print("[e14b] late_overlap reproduced for all rows (tolerance 1e-9)")

    j_self = np.array([jaccard(l, e) for l, e in zip(lates, early_sets)])
    yv = df_split.y.values

    def dist(x):
        return {"mean": round(float(np.mean(x)), 4), "median": round(float(np.median(x)), 4),
                "q25": round(float(np.percentile(x, 25)), 4),
                "q75": round(float(np.percentile(x, 75)), 4), "n": int(len(x))}

    # (2) among y=1: share of retained advisor concepts already occupied early
    shares = []
    for l, a, e, yy in zip(lates, adv_sets, early_sets, yv):
        if yy == 1:
            inter = l & a
            if inter:
                shares.append(len(inter & e) / len(inter))
    # (3) alternative label
    y_self = (j_self > SELF_THETA).astype(int)
    cm = pd.crosstab(pd.Series(yv, name="y"), pd.Series(y_self, name="y_self"))

    out = {
        "j_late_early_by_y": {"y=1": dist(j_self[yv == 1]), "y=0": dist(j_self[yv == 0])},
        "retained_adv_concepts_already_early_share_y1": {
            "mean": round(float(np.mean(shares)), 4),
            "median": round(float(np.median(shares)), 4), "n": len(shares)},
        "y_self": {
            "theta": SELF_THETA, "base_rate": round(float(y_self.mean()), 4),
            "y_base_rate": round(float(yv.mean()), 4),
            "confusion_matrix": {f"y={i}": {f"y_self={j}": int(cm.loc[i, j])
                                            for j in cm.columns} for i in cm.index},
            "cohen_kappa": round(float(cohen_kappa_score(yv, y_self)), 4),
            "agreement": round(float((yv == y_self).mean()), 4)},
    }

    # predict y_self vs y with the SAME full-tabular features (M2 logistic + M3 GBDT)
    Xt, _ = D.build_features(df_split, concepts="none")
    masks = {k: (df_split.split == k).values for k in ("train", "val", "test")}
    for label_name, lab in (("y", yv), ("y_self", y_self)):
        yb = {k: lab[m] for k, m in masks.items()}
        a = (Xt[masks["train"]], yb["train"], Xt[masks["val"]], yb["val"],
             Xt[masks["test"]], yb["test"])
        r2, _ = fit_logit(*a, seed=0)
        g_runs = [fit_gbdt(*a, seed=s)[0] for s in C.SEEDS]
        out.setdefault("predictability", {})[label_name] = {
            "M2_logit_tabular": {k: round(v, 4) for k, v in r2.items()},
            "M3_gbdt_tabular": S.summarize_seeds(g_runs)}
        print(f"[e14b] predict {label_name}: M2 PR={r2['auc_pr']:.4f}")
    return out


# ---------- part (c): E9e side split ----------

def run_side_ablation(df_split):
    combos = {
        "no_advisor": ["early_overlap", "adv_early_prod", "adv_early_breadth",
                       "adv_career_age_at_t0"],
        "advisor_only": ["early_overlap", "early_prod", "early_breadth",
                         "coauth_early_n", "coauth_early"],
        "no_coauth": ["coauth_early_n", "coauth_early"],
    }
    masks = {k: (df_split.split == k).values for k in ("train", "val", "test")}
    y = {k: df_split.loc[m, "y"].values for k, m in masks.items()}
    out = {}
    for name, drop in combos.items():
        X, cols = D.build_features(df_split, concepts="none", drop=drop)
        runs = [fit_gbdt(X[masks["train"]], y["train"], X[masks["val"]], y["val"],
                         X[masks["test"]], y["test"], seed=s)[0] for s in C.SEEDS]
        out[name] = {"features": cols, "summary": S.summarize_seeds(runs),
                     "per_seed_auc_pr": [round(r["auc_pr"], 4) for r in runs]}
        print(f"[e14c] {name}: AUC-PR {out[name]['summary']['auc_pr']['mean']:.4f} ({cols})")
    return out


def main():
    df = D.load_dataset()
    df_split = D.temporal_split(df)

    print(f"[e14] works store: {C.WORKS_STORE}")
    assert Path(C.WORKS_STORE).exists(), f"works store missing: {C.WORKS_STORE}"
    early_cnt, late_set = load_student_windows(df_split)

    F_student = student_features(df_split, early_cnt)
    X_st = student_tfidf(df_split)

    out = {"experiment": "E14_self_persistence", "seeds": C.SEEDS,
           "n": len(df_split), "works_store": str(C.WORKS_STORE)}
    out["a_student_only_ladder"] = run_ladder(df_split, F_student, X_st)
    out["b_label_decomposition"] = run_decomposition(df_split, late_set)
    out["c_side_ablation"] = run_side_ablation(df_split)

    (C.RESULTS_DIR / "e14_self_persistence.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["a_student_only_ladder"]["verdict"], indent=2))
    print("[e14] written:", C.RESULTS_DIR / "e14_self_persistence.json")


if __name__ == "__main__":
    main()
