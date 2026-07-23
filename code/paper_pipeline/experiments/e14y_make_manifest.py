"""E14y - export the M4b Colab manifest from the frozen table.

The SPECTER2 rung runs on a Colab GPU (M4b_SPECTER2_Colab.ipynb). It needs the
student ids, their t0 cutoffs, and their split assignment - but NOT the frozen
modeling table, which never leaves this machine. This script exports exactly that
minimum: one row per student, four columns.

    student_pid, st_openalex_id, t0, split

Output: <repo root>/m4b_manifest_<disc>.parquet  (econ, or neuro under the
NEURO_DATASET override), sitting next to the notebook that consumes it.

NOTE ON SHARED AUTHOR IDS. A few student rows resolve to the same OpenAlex author
id (econ: 15 ids shared by 31 of 6,043 rows). This is a property of the frozen
table, which is SHA-pinned and not repaired here. It matters downstream because a
SPECTER2 embedding is a property of the *author*, not the student row: the
notebook therefore pools over distinct ids and e14d_m4b.py joins many-to-one back
onto student rows. This script asserts the invariant that makes that join
well-defined - shared ids must agree on t0 (else the t0+5 encode cutoff is
ambiguous) and on split (else an author's works would straddle train and test,
leaking across folds). Both hold for econ as of 2026-07-15; the assert is here so
a future rebuild that breaks them fails loudly rather than silently.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C
from utils import data as D

SUFFIX = "neuro" if os.environ.get("NEURO_DATASET") else "econ"
OUT = C.REPO_ROOT.parents[2] / f"m4b_manifest_{SUFFIX}.parquet"

COLS = ["student_pid", "st_openalex_id", "t0", "split"]


def main():
    df = D.temporal_split(D.load_dataset())
    man = df[COLS].copy()

    chk = man.groupby("st_openalex_id").agg(t0n=("t0", "nunique"),
                                            spn=("split", "nunique"))
    bad_t0 = chk.index[chk.t0n > 1].tolist()
    bad_sp = chk.index[chk.spn > 1].tolist()
    assert not bad_t0, f"shared st_openalex_id disagrees on t0: {bad_t0[:5]}"
    assert not bad_sp, f"shared st_openalex_id straddles splits: {bad_sp[:5]}"

    n_rows, n_uniq = len(man), man.st_openalex_id.nunique()
    man.to_parquet(OUT, index=False)
    print(f"[e14y] wrote {OUT}")
    print(f"[e14y] {n_rows} student rows | {n_uniq} distinct authors "
          f"({n_rows - n_uniq} rows share an id with another student)")
    print(f"[e14y] splits: {man.split.value_counts().to_dict()}")
    print(f"[e14y] t0 range: {int(man.t0.min())}-{int(man.t0.max())}")
    print(f"[e14y] the Colab run must encode {n_uniq} authors; at the OpenAlex "
          f"free tier (~1000 fetches/day) budget {n_uniq / 1000:.0f}+ day(s) for "
          f"e14x_fetch_texts.py to supply their texts.")


if __name__ == "__main__":
    main()
