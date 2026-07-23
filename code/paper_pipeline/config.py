"""Central configuration - Who Inherits the Field?

Scope (with advisor, 2026-07): the frozen benchmark is economics, benchmarked to
depth. The pipeline is discipline-agnostic: a neuroscience replication runs the
same protocol on a second genealogy via the NEURO_DATASET override below (the
baseline ladder E1, the advisor-placebo control E10, and the standard-config HGT
E2/E12). The paper's first contribution is the leakage-free forward prediction
task + audit protocol. Decision rules are restated verbatim in each experiment script's docstring.

The frozen dataset must never be modified; every script verifies its SHA256.
"""
from pathlib import Path

# ---------- paths ----------
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT.parent / "data_econ"        # economics data (package layout)
CLEAN_DATASET = DATA_DIR / "clean_dataset.parquet"
PAIRS_FILE = DATA_DIR / "Econ_Pairs_With_Citations.parquet"   # full genealogy (E5 only)
OUTCOMES = DATA_DIR / "outcomes.parquet"
DATASET_SUMMARY = DATA_DIR / "dataset_summary_econ.json"
WORKS_STORE = DATA_DIR / "cache" / "works_store.jsonl"
# local checkout keeps the 199MB econ works store at the project root; the
# packaged layout keeps it under data_econ/cache. Same file, two locations.
_ROOT_WORKS_STORE = REPO_ROOT.parents[2] / "cache" / "works_store.jsonl"
if not WORKS_STORE.exists() and _ROOT_WORKS_STORE.exists():
    WORKS_STORE = _ROOT_WORKS_STORE
RESULTS_DIR = REPO_ROOT / "results_econ"   # economics results (renamed from results/)
FIGURES_DIR = REPO_ROOT / "figures_econ"   # economics figures (renamed from figures/)
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

EXPECTED_SHA256 = "5ef1eb6f4c061fc82d10839964deacc01fcf032f65d1dc67fd0e7aa06b97dba3"

# ---------- optional dataset override: neuroscience second-discipline replication ----------
# When NEURO_DATASET is set, the SAME experiment scripts run on the neuro dataset
# without disturbing the frozen econ config. Only the input file, its verified hash,
# and the output dirs move; every window/label/feature/seed parameter is unchanged
# (that identity is the identical-protocol claim). Econ behavior when env is unset.
# Hardcoded so the neuro run is genuinely hash-pinned (not self-computed).
NEURO_EXPECTED_SHA256 = "9e9dabb80c9ac9cbe5e996cfc1920cd46e77f5bcb0c30fe6710289745545ffa3"
import os as _os
if _os.environ.get("NEURO_DATASET"):
    CLEAN_DATASET = Path(_os.environ["NEURO_DATASET"])
    EXPECTED_SHA256 = _os.environ.get("NEURO_SHA256", NEURO_EXPECTED_SHA256)
    RESULTS_DIR = REPO_ROOT / "results_neuro"
    FIGURES_DIR = REPO_ROOT / "figures_neuro"
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    # outcomes + per-author works store follow the dataset (E6/E7/E14 on neuro);
    # sibling outcomes.parquet next to the overridden clean_dataset.parquet.
    OUTCOMES = CLEAN_DATASET.parent / "outcomes.parquet"
    WORKS_STORE = Path(_os.environ.get(
        "NEURO_WORKS_STORE",
        str(REPO_ROOT.parents[2] / "Neuro Data" / "build_neuro" / "cache" / "works_store.jsonl")))

# ---------- generic multidiscipline override (suite: econ/math/physics/chemistry/neuro) ----------
# Fires when DATASET is set and NEURO_DATASET is not. All five disciplines share one
# protocol on one 2026 snapshot; only the input file, its PINNED hash, and the output
# dirs move, and every window/label/feature/seed parameter is unchanged. SHAs are
# hardcoded here (auditable pin, not self-computed).
_PINNED_SHA = {
    "neuro": NEURO_EXPECTED_SHA256,
    # suite frozen tables, sha pinned as each build completes (auditable):
    "math": "aa5cc66ba9d284487b122c253fc5ab97376c99f3b5bd333bc256612cc6e02544",
    "physics": "9d3a4460c8f6dba069d222f5a4e2fb45cc822275e84c590b2fe156ce5e0c6d48",
    "chemistry": "0b4d21bf7b161b361a2e82904c4e55c2b17f5b928281177079b6a3fb060b8080",
    "neuro_min2": "14a9e823d63760cb29b06f258bd82a784a3b440eef2a8a205e76fd8db837ba07",
    "econ": "5ef1eb6f4c061fc82d10839964deacc01fcf032f65d1dc67fd0e7aa06b97dba3",
    # "econ_min2": "...":
}
_WORKS_STORE_OVERRIDE = {
    "neuro": REPO_ROOT.parents[2] / "Neuro Data" / "build_neuro" / "cache" / "works_store.jsonl",
    # >=2 arms reuse the parent discipline's frozen works store (built read-only):
    "neuro_min2": REPO_ROOT.parents[2] / "Neuro Data" / "build_neuro" / "cache" / "works_store.jsonl",
}
_ds = _os.environ.get("DATASET")
if _ds and not _os.environ.get("NEURO_DATASET"):
    _data_dir = REPO_ROOT.parent / f"data_{_ds}"
    CLEAN_DATASET = Path(_os.environ.get("DATASET_PATH", str(_data_dir / "clean_dataset.parquet")))
    EXPECTED_SHA256 = _os.environ.get("DATASET_SHA256") or _PINNED_SHA.get(_ds)
    if not EXPECTED_SHA256:
        raise SystemExit(f"config: DATASET={_ds} has no pinned SHA256 yet "
                         f"(build the frozen table first, or pass DATASET_SHA256).")
    RESULTS_DIR = REPO_ROOT / f"results_{_ds}"
    FIGURES_DIR = REPO_ROOT / f"figures_{_ds}"
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    OUTCOMES = CLEAN_DATASET.parent / "outcomes.parquet"
    _default_ws = _WORKS_STORE_OVERRIDE.get(
        _ds, REPO_ROOT.parents[2] / "suite_data" / _ds / "build" / "cache" / "works_store.jsonl")
    WORKS_STORE = Path(_os.environ.get("DATASET_WORKS_STORE", str(_default_ws)))

# ---------- temporal design (must match the dataset builder) ----------
EARLY_YEARS = 5
LATE_YEARS = 15
OBS_YEAR = 2026
JACCARD_THETA = 0.2
THETA_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]

# ---------- evaluation protocol ----------
SEEDS = list(range(10))
SPLIT_QUANTILES = (0.6, 0.8)
PRIMARY_METRIC = "auc_pr"
N_BOOTSTRAP = 2000

# ---------- feature policy ----------
TABULAR_FEATURES = [
    "early_overlap", "early_prod", "early_breadth",
    "adv_early_prod", "adv_early_breadth", "adv_career_age_at_t0",
    "coauth_early_n",
]
BOOL_FEATURES = ["coauth_early"]
LABEL = "y"
CONTINUOUS_LABEL = "late_overlap"

BANNED_COLUMNS = {
    "st_h_index", "adv_h_index", "adv_total_citations", "st_h_recomp",
    "co_authored", "st_cites_adv_broad", "adv_cites_st_broad", "career_len",
    "late_overlap", "late_prod", "y",
}

# ---------- house plotting style ----------
PALETTE = ["#3C5488", "#E64B35", "#00A087", "#4DBBD5", "#F39B7F",
           "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85"]
PALETTE_GREEN = ["#004949", "#20854E", "#009292", "#59A14F", "#8CD17D",
                 "#4E79A7", "#B6992D", "#499894", "#86BCB6", "#117733"]
EDGE_COLOR = "#BBBBBB"
EDGE_WIDTH = 0.5
LABEL_FONT = dict(size=10, color="#333333", family="Arial")
TITLE_FONT_SIZE = 18
