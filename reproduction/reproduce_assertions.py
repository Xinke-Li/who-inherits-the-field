#!/usr/bin/env python3
"""Assertion-style reproduction. Recomputes the headline numbers from the frozen
tables in ../data and checks them against the recorded SHA-256, base rates, and
label definition. Prints one line per check and exits non-zero on any failure.

Run from the package root:  python reproduction/reproduce_assertions.py
"""
import hashlib, json, os, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]
THETA = 0.2
fails = []
skipped = 0
n_run = 0


def check(name, cond):
    global n_run
    n_run += 1
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails.append(name)


# Scenario accounting for the three counts the paper prints. These constants
# must mirror the skip gates below: TABLE_GATED is every check that runs only
# when the five modeling tables are present (the per-field frozen checks, the
# sweep-versus-frozen base rates, the funnel year file), and WORKTREE_ONLY is
# every check that runs only in the author's tree (the two LaTeX-source checks,
# the concept-event parquet check, and the count check itself). A new gated
# check must update the matching constant or the count check at the end fails.
TABLE_GATED = 5 * 5 + 5 + 1
WORKTREE_ONLY = 2 + 1 + 1


# The five modeling tables are too large for git and ship in the Zenodo archive,
# so a bare clone has everything except them. Report that state instead of
# dying on the first missing file: the robustness checks below need no parquet
# and are the whole point of running this in a clone.
TABLES = {f: os.path.join(DATA, f"clean_dataset_{f}.parquet") for f in FIELDS}
HAVE_TABLES = all(os.path.exists(p) for p in TABLES.values())
if not HAVE_TABLES:
    missing = [f for f, p in TABLES.items() if not os.path.exists(p)]
    # 5 per discipline: one hash check plus four table-level checks. The
    # co-authorship axis check reads json only and still runs.
    skipped = 5 * len(FIELDS)
    print(f"[SKIP] {skipped} of the 26 frozen-artifact checks: "
          f"data/clean_dataset_{{{','.join(missing)}}}.parquet not present. "
          "The modeling tables ship in zenodo_archive.zip, not in the git "
          "clone; unpack it into data/ to run them.\n")

# 1. SHA-256 of every table matches SHA256SUMS
sums = {}
for line in open(os.path.join(DATA, "SHA256SUMS")):
    h, fn = line.split()
    sums[fn.lstrip("*").strip()] = h.strip()
if HAVE_TABLES:
    for f in FIELDS:
        fn = f"clean_dataset_{f}.parquet"
        got = hashlib.sha256(open(TABLES[f], "rb").read()).hexdigest()
        check(f"{f}: SHA-256 matches SHA256SUMS", got == sums.get(fn))

# 2. base rate and n_students match the summary; label equals (late_overlap > theta)
if HAVE_TABLES:
    for f in FIELDS:
        df = pd.read_parquet(TABLES[f])
        s = json.load(open(os.path.join(DATA, f"dataset_summary_{f}.json")))
        check(f"{f}: n_students matches summary ({len(df)})", len(df) == s["n_students"])
        check(f"{f}: base rate matches summary ({round(df.y.mean(),4)})",
              abs(df.y.mean() - s["label_base_rate"]) < 5e-4)
        check(f"{f}: label equals 1[late_overlap>{THETA}]",
              ((df.late_overlap > THETA).astype(int) == df.y).all())
        check(f"{f}: no right-censored rows (t0+15<=2026)", (df.t0 + 15 <= 2026).all())

# 3. co-authorship axis ordering: econ is the lowest early-window rate
axis = json.load(open(os.path.join(DATA, "coauthorship_axis.json")))["axis"]
rates = {k: v["early_window_coauth_openalex"] for k, v in axis.items()}
check("axis: econ has the lowest early-window coauth",
      rates["econ"] == min(rates.values()))

# 4. post-hoc robustness layer (revision-robustness branch): checked whenever
#    results/robustness ships with the artifact, so the audit property extends
#    to the new material. Absent directory = pre-robustness artifact, no checks.
ROB = os.path.join(HERE, "..", "results", "robustness")
if os.path.isdir(ROB):
    THETAS = ["0.10", "0.15", "0.20", "0.25", "0.30"]
    p = os.path.join(ROB, "theta_sweep_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        check("robustness: theta grid complete (5 fields x 5 thetas)",
              all(t in s["fields"][f]["rows"] for f in FIELDS for t in THETAS))
        # these five cross-reference the sweep against the frozen tables
        if HAVE_TABLES:
            for f in FIELDS:
                r = s["fields"][f]["rows"]["0.20"]
                df = pd.read_parquet(TABLES[f])
                check(f"robustness: {f} theta=0.20 base rate equals frozen",
                      abs(r["base_rate"] - df.y.mean()) < 5e-4)
        else:
            skipped += len(FIELDS)
            print(f"[SKIP] {len(FIELDS)} robustness checks that compare the "
                  "sweep's base rate against the frozen tables\n")
        for t in THETAS:
            m = json.load(open(os.path.join(ROB, f"theta_{t}.json")))
            check(f"robustness: theta_{t}.json has all fields, 10-seed ladders",
                  all(f in m and m[f]["ladder"]["M3_gbdt_tabular"]["auc_pr"]
                      ["n_seeds"] == 10 for f in FIELDS))
    p = os.path.join(ROB, "continuous_label_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        check("robustness: continuous-label check covers all fields, 10 seeds",
              all(f in s["fields"] and len(s["fields"][f]["per_seed"]
                  ["M3r_gbr_tabular"]) == 10 for f in FIELDS))
    p = os.path.join(ROB, "rgcn_symmetric_verdict.json")
    if os.path.exists(p):
        v = json.load(open(p))
        check("robustness: rgcn symmetric verdict carries all four models",
              set(v["models"]) == {"hgt", "hgt_tuned", "rgcn_symmetric",
                                   "gat_cohort_time"})
        check("robustness: rgcn grid selection recorded (16 configs)",
              len(v["grid_selection"]["all_configs_mean_val"]) == 16)
    p = os.path.join(ROB, "topk_sweep_summary.json")
    if os.path.exists(p):
        s = json.load(open(p))
        check("robustness: topk sweep complete (4 k-cells + 2 min-score per field)",
              all(len(s["fields"][f]["rows"]) == 6 for f in FIELDS))
        check("robustness: k=10 calibration reported for every field",
              all(f in s["calibration_k10"] for f in FIELDS))
    p = os.path.join(ROB, "e17_swap_label.json")
    if os.path.exists(p):
        s = json.load(open(p))
        check("robustness: swap-label control covers all fields, 10 redraws",
              all(f in s["fields"] and len(s["fields"][f]["per_seed"]) == 10
                  for f in FIELDS))
        check("robustness: swap-label exact calibration recorded per redraw",
              all("exact_theta_atom" in s["fields"][f]["per_seed"][0]
                  and "placebo_auc_roc" in s["fields"][f]["per_seed"][0]
                  for f in FIELDS))
        check("robustness: swap-label student-level CI present per field",
              all(len(s["fields"][f]["excess_ci95_student_pooled"]) == 2
                  for f in FIELDS))
    p = os.path.join(ROB, "power_tost.json")
    if os.path.exists(p):
        s = json.load(open(p))
        check("robustness: power/TOST covers 5 fields x 4 architectures",
              all(len(s["fields"][f]["models"]) == 4 for f in FIELDS))
    p = os.path.join(ROB, "topics_parallel.json")
    if os.path.exists(p):
        s = json.load(open(p))
        check("robustness: topics parallel covers 5 fields x 2 granularities",
              all(g in s["fields"][f] for f in FIELDS
                  for g in ("subfield", "topic")))
    g = os.path.join(ROB, "full_symmetric_grid")
    if os.path.isdir(g):
        check("robustness: symmetric grid has 10 winner cells x 10 seeds",
              all(os.path.exists(os.path.join(
                  g, f"{f}_{a}_sym_seed{s}.json"))
                  for f in FIELDS for a in ("rgcn", "gat") for s in range(10)))
        check("robustness: symmetric grid has 5 two-arch verdicts",
              all(set(json.load(open(os.path.join(g, f"{f}_verdict.json")))
                      ["models"]) == {"rgcn_sym", "gat_sym"} for f in FIELDS))
        check("robustness: symmetric grid DONE flag present",
              os.path.exists(os.path.join(g, "DONE_fullgrid.flag")))
    p = os.path.join(ROB, "extra_rungs.json")
    if os.path.exists(p):
        s = json.load(open(p))
        check("robustness: extra rungs (catboost/tabpfn) cover all fields",
              all(f in s["fields"] for f in FIELDS))
    p = os.path.join(ROB, "SHA256SUMS.robustness")
    if os.path.exists(p):
        n_bad = 0
        for line in open(p):
            h, fn = line.split(None, 1)
            fp = os.path.join(ROB, fn.strip())
            if not os.path.exists(fp) or hashlib.sha256(
                    open(fp, "rb").read()).hexdigest() != h:
                n_bad += 1
        check("robustness: SHA256SUMS.robustness verifies", n_bad == 0)

# ---------------------------------------------------------------------------
# T5.1: the revision layer. The integrity logic is r28_assemble_tables.py's,
# ported here so a clone checks it without running the assembler. A sweep bug
# once wrote economics outputs into chemistry's directory and nothing caught it
# until the numbers were read by eye; these three properties find that class of
# fault in seconds.
# ---------------------------------------------------------------------------
REV = os.path.join(HERE, "..", "results", "revision")
SUP = os.path.join(HERE, "..", "data", "supplement")
STRICT_TREE = os.path.join(REV, "T2_1_strict_contract", "T2_1_final",
                           "T2_1_strict_contract")
LINEAGE_TREE = os.path.join(REV, "T2_2b_lineage_contract")
M5P = {"chemistry": 0.5374, "econ": 0.3613, "math": 0.4357,
       "neuro": 0.4249, "physics": 0.6485}
NTEST = {"econ": 495, "math": 935, "neuro": 3463, "physics": 2218,
         "chemistry": 4617}
CELL_RE = __import__("re").compile(r"^(rgcn|gat|hgt)_(prereg|tuned)_(strict|legacy)$")
rev_skipped = 0


def verdict_cells(tree, cell_re):
    """(<field>, <cell>) -> summary dict, for directories that name a verdict."""
    out = {}
    for f in FIELDS:
        d = os.path.join(tree, f)
        if not os.path.isdir(d):
            continue
        for c in sorted(os.listdir(d)):
            p = os.path.join(d, c, "summary.json")
            if cell_re.match(c) and os.path.exists(p):
                out[(f, c)] = json.load(open(p, encoding="utf-8"))
    return out


if os.path.isdir(STRICT_TREE):
    cells = verdict_cells(STRICT_TREE, CELL_RE)
    check(f"revision: 39 strict verdict cells present ({len(cells)})",
          len(cells) == 39)
    bad_field = [k for k, v in cells.items() if v.get("field") != k[0]]
    check("revision: every strict cell's field matches its directory",
          not bad_field)
    bad_ceil = [k for k, v in cells.items()
                if abs(float(v.get("M5prime_mean", -9)) - M5P[k[0]]) > 1e-3]
    check("revision: every strict cell's M5' matches the frozen ceiling",
          not bad_ceil)
    bad_len, bad_seeds = [], []
    for (f, c), v in cells.items():
        d = os.path.join(STRICT_TREE, f, c)
        seeds = [os.path.join(d, f"seed{s}.json") for s in range(10)]
        if not all(os.path.exists(p) for p in seeds):
            bad_seeds.append((f, c))
            continue
        for p in seeds:
            j = json.load(open(p, encoding="utf-8"))
            if (len(j.get("test_scores") or []) != NTEST[f]
                    or len(j.get("test_labels") or []) != NTEST[f]):
                bad_len.append((f, c))
                break
    check("revision: every strict cell has ten per-seed files (F8a)",
          not bad_seeds)
    check("revision: every per-seed score array matches its cohort size",
          not bad_len)
    routed = {v.get("target_table") for v in cells.values()}
    check("revision: every strict cell declares a target table",
          routed == {"Table 12b", "Table 13b"})

    p = os.path.join(STRICT_TREE, "assembled_tables.json")
    if os.path.exists(p):
        a = json.load(open(p, encoding="utf-8"))
        check("revision: assembled tables report 39 cells, 0 violations",
              a["integrity"]["cells"] == 39 and a["integrity"]["violations"] == 0)
        check("revision: assembled row counts are 20 and 19",
              a["tables"]["Table 12b"]["n_rows"] == 20
              and a["tables"]["Table 13b"]["n_rows"] == 19)
        check("revision: five assembled cells exceed",
              sum(a["tables"][t]["n_exceeds"] for t in a["tables"]) == 5)

    p = os.path.join(STRICT_TREE, "chemistry", "attribution", "attribution.json")
    if os.path.exists(p):
        at = json.load(open(p, encoding="utf-8"))
        check("revision: the five attribution cells share one session",
              bool(at.get("same_session")) and at.get("session_id"))
        m = at["cell_mean_auc_pr"]
        check("revision: attribution deltas agree with the cell means",
              abs(round(m["B"] - m["A"], 4)
                  - at["F2_class_weight_B_minus_A"]) < 1e-9
              and abs(round(m["E"] - m["B"], 4)
                      - at["F1a_advisor_keying_E_minus_B"]) < 1e-9
              and abs(round(m["C"] - m["B"], 4)
                      - at["F1_both_C_minus_B"]) < 1e-9)
        check("revision: the additivity gap sits inside the determinism floor",
              at["additivity_gap"] <= at["determinism_floor"])
    else:
        rev_skipped += 3
        print("[SKIP] 3 attribution checks: run "
              "code/r27_attribution.py --verify-only --dir <attribution tree>")
else:
    rev_skipped += 9
    print(f"[SKIP] 9 strict-contract checks: {STRICT_TREE} not present")

# The lineage arm is a GPU leg. Absent until it lands, and reported as absent
# rather than silently passing.
#
# The test is whether a verdict cell exists, not whether the directory does.
# A --stage gate1 invocation creates <field>/<cell>/ and writes gate1.json into
# it without training anything, so a directory-only test reports the leg as
# landed and then fails every check inside it. That happened.
LINEAGE_LANDED = bool(__import__("glob").glob(
    os.path.join(LINEAGE_TREE, "*", "*", "summary.json")))
if LINEAGE_LANDED:
    lre = __import__("re").compile(r"^(rgcn|gat)_(prereg|tuned)_lineage$")
    lin = verdict_cells(LINEAGE_TREE, lre)
    check(f"revision: 20 lineage verdict cells present ({len(lin)})",
          len(lin) == 20)
    check("revision: every lineage cell's field matches its directory",
          not [k for k, v in lin.items() if v.get("field") != k[0]])
    check("revision: every lineage cell's M5' matches the frozen ceiling",
          not [k for k, v in lin.items()
               if abs(float(v.get("M5prime_mean", -9)) - M5P[k[0]]) > 1e-3])
    check("revision: every lineage cell names its matching strict cell",
          all((v.get("matching_strict_cell") or {}).get("found")
              for v in lin.values()))
    check("revision: every lineage cell carries ancestry edges",
          all((v.get("lineage_graph") or {}).get("ancestry_edges", 0) > 0
              for v in lin.values()))
    check("revision: lineage cells route to Tables 12c and 13c",
          {v.get("target_table") for v in lin.values()}
          == {"Table 12c", "Table 13c"})
else:
    rev_skipped += 6
    print("[SKIP] 6 lineage checks: the T2.2b GPU leg has not landed yet")

# supplement manifests and the time-contract verification
p = os.path.join(SUP, "concept_events_manifest.json")
if os.path.exists(p):
    man = json.load(open(p, encoding="utf-8"))
    check("supplement: concept events cover all five disciplines",
          all(f in man for f in FIELDS))
    # This check runs only in the author's working tree, gated on the same
    # paper/main.tex marker the count assertion uses. In a clone it skips:
    # deposit completeness is pinned by DEPOSIT_MANIFEST.json and verified at
    # release time by r43, so a clone run does not re-check it.
    _worktree = os.path.isfile(os.path.join(HERE, os.pardir, "paper",
                                            "main.tex"))
    if _worktree and all(
            os.path.exists(os.path.join(SUP, f"concept_events_{f}.parquet"))
            for f in FIELDS):
        check("supplement: every concept event table exists", True)
    else:
        rev_skipped += 1
        print("[SKIP] 1 concept-event table check: it runs in the author's "
              "working tree only; deposit completeness is pinned by "
              "DEPOSIT_MANIFEST.json and verified at release, so a clone "
              "does not re-check it")
    check("supplement: concept events restrict to the builder's concept view",
          all(man[f]["concept_min_score"] == 0.3
              and man[f]["work_top_concepts"] == 3 for f in FIELDS))
else:
    rev_skipped += 3
    print("[SKIP] 3 concept-event checks: run code/r36_concept_events.py")

p = os.path.join(REV, "T2_11_time_contract", "verification.json")
if os.path.exists(p):
    ver = json.load(open(p, encoding="utf-8"))
    check("supplement: the t0+5 assertion is exact in all five disciplines",
          bool(ver.get("all_exact_assertions_pass"))
          and all(v["exact_t0plus5_assertion"]["violations"] == 0
                  for v in ver["fields"].values())
          and set(ver["fields"]) == set(FIELDS))
    check("supplement: agreement with the frozen columns is inside its band",
          bool(ver.get("all_agreement_in_expected_band")))
else:
    rev_skipped += 2
    print("[SKIP] 2 time-contract checks: run reproduction/verify_time_contract.py")

p = os.path.join(SUP, "lineage_manifest.json")
if os.path.exists(p):
    man = json.load(open(p, encoding="utf-8"))
    check("supplement: lineage manifest covers all five disciplines",
          all(f in man for f in FIELDS))
    if all(os.path.exists(os.path.join(SUP, f"lineage_{f}.parquet"))
           for f in FIELDS):
        check("supplement: every lineage table exists", True)
    else:
        rev_skipped += 1
        print("[SKIP] 1 lineage-table check: the lineage_*.parquet supplement "
              "is not in the git clone; see the datasheet for where it is "
              "deposited")
    check("supplement: lineage profiles use the builder's concept view",
          all(man[f].get("concept_min_score") == 0.3
              and man[f].get("work_top_concepts") == 3 for f in FIELDS))
else:
    rev_skipped += 2
    print("[SKIP] 2 lineage-table checks: run code/r31_lineage_table.py")

p = os.path.join(REV, "T2_8_coverage_sensitivity", "strata.json")
if os.path.exists(p):
    st = json.load(open(p, encoding="utf-8"))
    check("revision: economics has the lowest early co-authorship in all three "
          "productivity strata",
          bool(st.get("econ_lowest_in_all_strata"))
          and all(st["ordering_within_strata"][k]["econ_is_lowest"]
                  for k in ("low", "mid", "high")))
else:
    rev_skipped += 1
    print("[SKIP] 1 co-authorship stratum check: run code/r30_coauth_strata.py")

p = os.path.join(REV, "T2_8_coverage_sensitivity", "coverage_strata.json")
if os.path.exists(p):
    cs = json.load(open(p, encoding="utf-8"))
    # the second stratifier the objection named: index coverage, not
    # productivity. Both arms must agree, or the ordering rests on where the
    # noisy small-institution densities happened to fall
    check("revision: economics stays lowest on early co-authorship in every "
          "coverage tercile, in both institution arms",
          all(a["econ_lowest_in_all_strata"] for a in cs["arms"].values())
          and len(cs["arms"]) == 2)
else:
    rev_skipped += 1
    print("[SKIP] 1 coverage stratum check: run code/r54_coverage_strata.py")

p = os.path.join(REV, "T3_3_mechanism", "verdict.json")
if os.path.exists(p):
    mv = json.load(open(p, encoding="utf-8"))
    check("revision: T3.3 reports all three mechanism candidates",
          all(k in mv for k in ("candidate_1_cohort_base_rate_dispersion",
                                "candidate_2_concept_graph_density",
                                "candidate_3_vocabulary_granularity")))
    check("revision: T3.3 records that the three anomalies do not move together",
          mv["candidate_1_cohort_base_rate_dispersion"]
            ["all_three_attenuate_together"] is False)
else:
    rev_skipped += 2
    print("[SKIP] 2 mechanism checks: run code/r38_t33_mechanism.py")

p = os.path.join(REV, "T2_7_minworks", "e16_minworks_sensitivity.json")
if os.path.exists(p):
    mw = json.load(open(p, encoding="utf-8"))
    m3 = mw["min3_arm_regenerated"]
    check("revision: T2.7 regenerates the frozen neuroscience gap and rung the "
          "appendix quotes (0.111 and 0.426)",
          m3["advisor_placebo_gap"]["match"] and m3["best_tabular_rung"]["match"])
    check("revision: T2.7 records the relaxed table as absent rather than "
          "adopting the two numbers that need it",
          mw["min2_arm"]["table_present"] is False
          and set(mw["min2_arm"]["not_regenerable"])
          >= {"advisor_placebo_gap", "best_tabular_rung"})
else:
    rev_skipped += 2
    print("[SKIP] 2 min-works checks: run code/r50_t27_minworks.py")

_conc = [os.path.join(HERE, os.pardir, "data", "supplement",
                      f"early_concentration_{f}.parquet") for f in FIELDS]
if all(os.path.exists(q) for q in _conc):
    import pandas as _pd
    _ok, _rows = True, {}
    for f, q in zip(FIELDS, _conc):
        d = _pd.read_parquet(q)
        _rows[f] = len(d)
        _ok &= (list(d.columns) == ["student_pid", "early_concentration"]
                and d.early_concentration.between(0, 1).all()
                and d.student_pid.is_unique)
    check("revision: T2.4 supplies early_concentration for all five "
          f"disciplines, keyed and in range ({sum(_rows.values())} rows)", _ok)
else:
    rev_skipped += 1
    print("[SKIP] 1 concentration check: run code/r47_early_concentration.py")

p = os.path.join(REV, "T2_4_e14_full5", "full5_floor_summary.json")
if os.path.exists(p):
    f5 = json.load(open(p, encoding="utf-8"))
    cal = f5.get("calibration_vs_frozen_e14", {})
    # the reconstruction is judged against the frozen certificate it rebuilds,
    # not asserted to be it: 0.005 AUC-PR is four times the 0.0013 determinism
    # floor and an order below the effect the fifth feature is being read for
    check(f"revision: T2.4's rebuilt fifth feature reproduces the frozen e14 "
          f"floor in every calibrated discipline ({len(cal)})",
          bool(cal) and all(abs(v["floor_delta"]) <= 0.005 for v in cal.values()))
    _cells = sum(b["n_cells"] for b in f5["fields"].values())
    _moved = {f for f, b in f5["fields"].items() if b["n_branch_moves"]}
    check(f"revision: T2.4 covers eleven cells in each of five disciplines "
          f"({_cells})",
          _cells == 55 and all(b["n_cells"] == 11 for b in f5["fields"].values()))
    # the paper's claim: the fifth feature moves mathematics and nothing else
    check("revision: mathematics is the only discipline whose branch the fifth "
          "feature moves",
          _moved == {"math"} and f5["fields"]["math"]["n_branch_moves"] == 5)
else:
    rev_skipped += 1
    print("[SKIP] 1 full-five calibration check: run "
          "code/r48_full5_floor.py --merge")

p = os.path.join(HERE, os.pardir, "data", "supplement",
                 "funnel_neuro_complete.json")
if os.path.exists(p):
    fn = json.load(open(p, encoding="utf-8"))
    rec = fn["reconstructed"]
    check("supplement: the neuroscience funnel's eight uninstrumented counters "
          "are reconstructed and none is left at zero",
          rec["both_in_works_store"] > 0 and rec["survive_window_filters"] > 0
          and len(rec["drop_breakdown"]) == 6
          and sum(rec["drop_breakdown"].values()) > 0)
    check("supplement: the reconstructed funnel keeps the recorded counters "
          "beside it and reproduces the two that were instrumented",
          fn["agreement"]["raw_pairs"]["match"]
          and fn["agreement"]["both_resolvable"]["match"]
          and fn["recorded"]["survive_window_filters"] == 0)
    # The counters are a function of publication years alone, so the 287 MB
    # fetch cache they came from does not have to ship for a reader to check
    # them. This 2.2 MB file is what r49 --stage verify recomputes all eight
    # from; without it the reconstruction would be a number to take on trust.
    _yp = os.path.join(HERE, os.pardir, "data", "supplement",
                       "funnel_neuro_years.parquet")
    if os.path.exists(_yp) and HAVE_TABLES:
        import pandas as _pd
        _yd = _pd.read_parquet(_yp)
        check(f"supplement: the funnel year file covers every author of the "
              f"resolvable neuroscience pairs ({len(_yd)})",
              list(_yd.columns) == ["aid", "n_works", "years"]
              and len(_yd) == 64476 and _yd.aid.is_unique)
    else:
        rev_skipped += 1
        print("[SKIP] 1 funnel year-file check: run "
              "code/r49_funnel_neuro_counters.py --stage years")
else:
    rev_skipped += 2
    print("[SKIP] 2 funnel-counter checks: run "
          "code/r49_funnel_neuro_counters.py --stage count")

# ---- escape-mangling scan over the LaTeX sources ----
# A backslash command that passes through a shell heredoc or an unescaped
# Python string loses its backslash to the control character it names: \ref
# becomes a carriage return and the orphan "ef". LaTeX typesets the orphan
# literally and reports nothing, so a build shows zero undefined references
# while printing a label key into the PDF. One instance reached a built PDF in
# this project before it was found by hand. This makes the scan standing, so a
# regenerated appendix fragment cannot reintroduce it silently.
_paper = os.path.join(HERE, os.pardir, "paper")
if not os.path.isdir(_paper):
    rev_skipped += 1
    print("[SKIP] 1 LaTeX escape-mangling check: paper/ is not part of the "
          "public release, so there are no .tex sources to scan here")
else:
    sys.path.insert(0, os.path.join(HERE, os.pardir, "code"))
    try:
        import r44_escape_scan as _r44
    except ImportError:
        check("revision: code/r44_escape_scan.py is present to scan the "
              "LaTeX sources", False)
    else:
        _real, _susp, _files = _r44.run()
        for _h in _real + _susp:
            print(f"   {_h[0]}:{_h[1]}:{_h[2]}  {_h[3]!r}  {_h[5]}")
        check(f"revision: no mangled backslash command in the {len(_files)} "
              f"LaTeX sources the paper builds",
              not _real and not _susp)
        # a caption is a moving argument, so \path, \url and \verb must be
        # \protect-ed there; unprotected they raise "\url used in a moving
        # argument", and the error names \url whatever the source said
        _frag = _r44.fragile_in_captions()
        for _h in _frag:
            print(f"   {_h[0]}:{_h[1]}  unprotected {_h[2]} in a caption")
        check("revision: every fragile command inside a caption is protected",
              not _frag)

# The counts the paper and README print for this harness drifted three times
# while they were kept by hand, so the harness now checks every location that
# prints one. This run's executed-check count is the working-tree number; the
# archive and bare-clone numbers derive from it through the same constants the
# skip gates use. The check needs the working tree, the one place every other
# check runs, so it skips wherever any other check has skipped.
def _tens_word(n):
    tens = {2: "Twenty", 3: "Thirty", 4: "Forty", 5: "Fifty", 6: "Sixty",
            7: "Seventy", 8: "Eighty", 9: "Ninety"}
    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine"]
    if not 20 <= n <= 99:
        return None
    w = tens[n // 10]
    return w + ("-" + ones[n % 10] if n % 10 else "")


_main_tex = os.path.join(HERE, os.pardir, "paper", "main.tex")
_readme = os.path.join(HERE, os.pardir, "README.md")
if not os.path.isfile(_main_tex) or skipped or rev_skipped:
    rev_skipped += 1
    print("[SKIP] 1 check-count check: it needs paper/main.tex and a tree "
          "where no other check has skipped")
else:
    _src = open(_main_tex, encoding="utf-8").read()
    _W = n_run + 1                      # this check is part of the count
    _N = _W - WORKTREE_ONLY             # release tree with the archive
    _M = _N - TABLE_GATED               # release tree without the archive
    _m1 = re.search(r"runs (\d+) checks in four groups", _src)
    _m2 = re.search(r"it runs (\d+) and skips the (\d+) that read", _src)
    _m3 = re.search(r"working tree it runs (\d+)", _src)
    _got = [(int(_m1.group(1)) if _m1 else None, _N, "main.tex archive count"),
            (int(_m2.group(1)) if _m2 else None, _M, "main.tex bare count"),
            (int(_m2.group(2)) if _m2 else None, TABLE_GATED,
             "main.tex table-gated count"),
            (int(_m3.group(1)) if _m3 else None, _W, "main.tex working tree")]
    if os.path.isfile(_readme):
        _rd = open(_readme, encoding="utf-8").read()
        _r1 = re.search(r"harness runs \*\*(\d+) checks\*\*", _rd)
        _r2 = re.search(r"it runs (\d+) and skips the\s+(\d+) that read", _rd)
        _r3 = re.search(r"(\w+(?:-\w+)?) of those checks read the five "
                        r"modeling tables", _rd)
        _r4 = re.search(r"runs the other (\d+); unpack the archive into "
                        r"`data/` for all (\d+)", _rd)
        _got += [(int(_r1.group(1)) if _r1 else None, _N, "README archive"),
                 (int(_r2.group(1)) if _r2 else None, _M, "README bare"),
                 (int(_r2.group(2)) if _r2 else None, TABLE_GATED,
                  "README table-gated"),
                 (_r3.group(1) if _r3 else None, _tens_word(TABLE_GATED),
                  "README table-gated, word form"),
                 (int(_r4.group(1)) if _r4 else None, _M, "README bare again"),
                 (int(_r4.group(2)) if _r4 else None, _N, "README archive again")]
    _bad = [f"{w}: printed {g!r}, harness {e!r}" for g, e, w in _got if g != e]
    for _b in _bad:
        print(f"   {_b}")
    check(f"paper: every check count main.tex and README print equals the "
          f"harness's own accounting (archive {_N}, bare {_M}, "
          f"working tree {_W})", not _bad)

print()
if fails:
    print(f"{len(fails)} check(s) FAILED: {fails}")
    sys.exit(1)
total_skipped = skipped + rev_skipped
if total_skipped:
    print(f"all reproduction checks that could run passed; {total_skipped} "
          f"skipped ({skipped} for want of the modeling tables, {rev_skipped} "
          f"for artifacts a GPU leg or a fetch has not produced yet; see the "
          f"SKIP lines above)")
else:
    print("all reproduction checks passed")
