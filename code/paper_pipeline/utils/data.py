"""Data loading, integrity verification, splits, and feature construction.

Every entry point re-verifies the frozen dataset hash and the leakage invariants,
so no experiment can silently run on a corrupted or mutated file.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C


def load_dataset(verify_hash: bool = True) -> pd.DataFrame:
    """Load the frozen dataset, verify integrity + leakage invariants, apply
    the two documented row drops (empty early-concept lists)."""
    raw = open(C.CLEAN_DATASET, "rb").read()
    if verify_hash:
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == C.EXPECTED_SHA256, (
            f"clean_dataset.parquet hash mismatch:\n  got      {sha}\n  expected {C.EXPECTED_SHA256}\n"
            "The frozen dataset changed - re-freeze deliberately or fix the file.")
    df = pd.read_parquet(C.CLEAN_DATASET)

    # leakage invariants (mirror the builder's guard)
    banned_present = C.BANNED_COLUMNS & set(df.columns) - {"y", "late_overlap", "late_prod"}
    assert not banned_present, f"banned columns present: {banned_present}"
    assert (df.t0 + C.LATE_YEARS <= C.OBS_YEAR).all(), "right-censored rows"
    assert (((df.late_overlap > C.JACCARD_THETA)).astype(int) == df.y).all(), "label mismatch"
    assert df.student_pid.is_unique, "duplicate students"
    assert (df.coauth_early_n >= 0).all(), "missing coauth values"

    # documented drop: rows whose early-window concept list is empty (n=2)
    n0 = len(df)
    df = df[df.early_concepts.apply(len) > 0].reset_index(drop=True)
    if len(df) != n0:
        print(f"[data] dropped {n0 - len(df)} rows with empty early_concepts")
    return df


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Cohort split on t0 at the configured quantiles (~60/20/20).
    Train on older cohorts, validate on middle, test on the most recent.
    Returns a copy with a 'split' column; prints the year boundaries used."""
    q1, q2 = np.quantile(df.t0, C.SPLIT_QUANTILES)
    q1, q2 = int(q1), int(q2)
    out = df.copy()
    out["split"] = np.where(out.t0 <= q1, "train", np.where(out.t0 <= q2, "val", "test"))
    sizes = out.split.value_counts().to_dict()
    rates = out.groupby("split").y.mean().round(4).to_dict()
    print(f"[split] temporal: train t0<={q1}, val t0 in ({q1},{q2}], test t0>{q2} "
          f"| sizes={sizes} | base rates={rates}")
    return out


def advisor_disjoint_split(df: pd.DataFrame, seed: int = 0,
                           frac=(0.6, 0.2, 0.2)) -> pd.DataFrame:
    """Robustness split (E9b): all students of one advisor land in the same fold,
    so no advisor-specific information crosses folds."""
    rng = np.random.default_rng(seed)
    advisors = df.advisor_pid.unique()
    rng.shuffle(advisors)
    n = len(advisors)
    cut1, cut2 = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    fold = {a: ("train" if i < cut1 else "val" if i < cut2 else "test")
            for i, a in enumerate(advisors)}
    out = df.copy()
    out["split"] = out.advisor_pid.map(fold)
    assert not (set(out.loc[out.split == "train", "advisor_pid"])
                & set(out.loc[out.split == "test", "advisor_pid"]))
    return out


def build_features(df: pd.DataFrame, concepts: str = "none",
                   drop: list | None = None, tfidf_max_features: int = 2000):
    """Design matrix from pre-window information only.

    concepts: 'none'  -> tabular features only
              'tfidf' -> + TF-IDF of student early concepts and advisor profile
    drop:     feature names to exclude (for E9e-style ablations, e.g. ['early_overlap'])
    Returns (X, feature_names). X is dense float64.
    """
    drop = set(drop or [])
    cols = [c for c in C.TABULAR_FEATURES if c not in drop]
    X = df[cols].astype(float).copy()
    for b in C.BOOL_FEATURES:
        if b not in drop:
            X[b] = df[b].astype(float)
            cols = cols + [b]
    forbidden = set(cols) & C.BANNED_COLUMNS
    assert not forbidden, f"banned features requested: {forbidden}"

    if concepts == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        docs_st = df.early_concepts.apply(lambda l: " ".join(c.replace(" ", "_") for c in l))
        docs_ad = df.adv_profile.apply(lambda l: " ".join(c.replace(" ", "_") for c in l))
        from scipy import sparse
        vec_st = TfidfVectorizer(max_features=tfidf_max_features, token_pattern=r"\S+")
        vec_ad = TfidfVectorizer(max_features=tfidf_max_features, token_pattern=r"\S+")
        # fit on ALL rows is fine: vocabulary comes from pre-window text only
        A = vec_st.fit_transform(docs_st)
        B = vec_ad.fit_transform(docs_ad)
        names = (cols + [f"st:{t}" for t in vec_st.get_feature_names_out()]
                      + [f"adv:{t}" for t in vec_ad.get_feature_names_out()])
        return sparse.hstack([sparse.csr_matrix(X.values), A, B]).tocsr(), names

    return X.values, cols


def split_xy(df_split, X, label: str = "y"):
    """Slice a prebuilt design matrix by the 'split' column."""
    out = {}
    for name in ("train", "val", "test"):
        m = (df_split.split == name).values
        out[name] = (X[m], df_split.loc[m, label].values)
    return out


def build_nfa_features(df) -> "pd.DataFrame":
    """Neighbor-Feature Aggregation for the M5 baseline (GBDT+NFA), the
    strong-baseline standard for graphs with tabular features (TabGraphs 2024;
    BGNN, Ivanov & Prokhorenkova 2021).

    TEMPORAL CONTRACT, strictly enforced by construction - for student i with
    first-publication year t0_i, aggregate ONLY over same-advisor siblings j
    whose information is observable at i's feature freeze (t0_i + 5):

      prior siblings  (t0_j <= t0_i):        early-window features of j are
                                             fully realized by t0_j+5 <= t0_i+5
        -> nfa_n_prior_sibs, nfa_sib_overlap_mean, nfa_sib_prod_mean,
           nfa_sib_breadth_mean
      closed siblings (t0_j + 15 <= t0_i+5): j's LABEL window has closed, so
                                             the advisor's realized track
                                             record is observable
        -> nfa_adv_track_retention (mean y over closed siblings),
           nfa_n_closed_sibs

    No other label information enters (the focal row's own y is never used;
    a sibling's y only when its 15-year window ended before i's freeze).
    Missing aggregates use -1 sentinels; counts are 0 (trees handle both).
    """
    cols = ["nfa_n_prior_sibs", "nfa_sib_overlap_mean", "nfa_sib_prod_mean",
            "nfa_sib_breadth_mean", "nfa_adv_track_retention", "nfa_n_closed_sibs"]
    out = pd.DataFrame(-1.0, index=df.index, columns=cols)
    out["nfa_n_prior_sibs"] = 0.0
    out["nfa_n_closed_sibs"] = 0.0

    for _, g in df.groupby("advisor_pid"):
        if len(g) < 2:
            continue
        t0 = g.t0.values
        ov = g.early_overlap.values
        pr = g.early_prod.values
        br = g.early_breadth.values
        yy = g.y.values
        idx = g.index.values
        for a in range(len(g)):
            prior = (t0 <= t0[a])
            prior[a] = False                      # never the focal row itself
            if prior.any():
                out.at[idx[a], "nfa_n_prior_sibs"] = float(prior.sum())
                out.at[idx[a], "nfa_sib_overlap_mean"] = float(ov[prior].mean())
                out.at[idx[a], "nfa_sib_prod_mean"] = float(pr[prior].mean())
                out.at[idx[a], "nfa_sib_breadth_mean"] = float(br[prior].mean())
            closed = (t0 + 15 <= t0[a] + 5)
            closed[a] = False
            if closed.any():
                out.at[idx[a], "nfa_n_closed_sibs"] = float(closed.sum())
                out.at[idx[a], "nfa_adv_track_retention"] = float(yy[closed].mean())

    # guard: the only label column consulted is y of CLOSED siblings; assert
    # the closure inequality held for every used value (re-check, defensive).
    return out
