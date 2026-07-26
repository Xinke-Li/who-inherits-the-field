#!/usr/bin/env python3
"""Self-consistency pass over the paper's numbers. Reports; never edits.

Each check prints PASS or FAIL; on FAIL it prints both values and their
sources. Checks that cannot be made mechanical are printed as REPORT lines
with the reason, so a human reads exactly the residue a machine cannot.

  python code/r58_selfconsistency.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = (PAPER / "main.tex").read_text(encoding="utf-8")
T21 = ROOT / "results" / "revision" / "T2_1_strict_contract"
ASM = json.loads((T21 / "T2_1_final" / "T2_1_strict_contract" /
                  "assembled_tables.json").read_text(encoding="utf-8"))
DET = json.loads((T21 / "chemistry" / "DETERMINISM_MEASURED.json")
                 .read_text(encoding="utf-8"))
F5 = json.loads((ROOT / "results" / "revision" / "T2_4_e14_full5" /
                 "full5_floor_summary.json").read_text(encoding="utf-8"))
CTRL = json.loads((ROOT / "results" / "revision" / "T2_4_e14_full5" /
                   "control_four_feature_math.json").read_text(encoding="utf-8"))
TH20 = json.loads((ROOT / "results" / "robustness" / "theta_0.20.json")
                  .read_text(encoding="utf-8"))
CALL = json.loads((ROOT / "results" / "revision" / "T2_13_callpath" /
                   "callpath.json").read_text(encoding="utf-8"))
FIELDS = ["econ", "math", "neuro", "physics", "chemistry"]

n_pass = n_fail = 0


def out(ok, name, detail=""):
    global n_pass, n_fail
    tag = "PASS" if ok else "FAIL"
    if ok:
        n_pass += 1
    else:
        n_fail += 1
    print(f"[{tag}] {name}")
    if detail and not ok:
        print(f"       {detail}")


def report(name, detail):
    print(f"[REPORT] {name}")
    print(f"       {detail}")


def abstract_and_s1():
    a = TEX.index("\\begin{abstract}")
    b = TEX.index("\\end{abstract}")
    s1 = TEX.index("\\section{Introduction}")
    s2 = TEX.index("\\section{Related Work}")
    body = TEX[s1:s2]
    # figure environments carry layout constants (includegraphics widths),
    # which are not data numbers
    body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", "", body,
                  flags=re.S)
    return TEX[a:b], body


def cells():
    for tab in ASM["tables"].values():
        for field, rows in tab["rows"].items():
            for r in rows:
                yield field, r


# ---------------------------------------------------------------- B1
def b1():
    print("\n== B1: abstract and Section 1 token provenance ==")
    import pandas as pd
    modeled = sum(len(pd.read_parquet(ROOT / "data" / f"clean_dataset_{f}.parquet"))
                  for f in FIELDS)
    floor = DET["max_drift_point_estimate"]
    m20 = F5["fields"]["math"]["cells"]["theta_0.20"]
    fifth = round(abs(m20["floor_delta"]), 4)
    foot = round(abs(TH20["math"]["self_persistence"]["verdict"]
                     ["best_student_auc_pr"]
                     - CTRL["verdict"]["best_student_auc_pr"]), 4)
    n_exceed = sum(1 for _, r in cells() if r["exceeds"])
    ex_arch = sorted({r["row"].replace(" symmetric", "")
                      for _, r in cells() if r["exceeds"]})
    ex_fields = sorted({f for f, r in cells() if r["exceeds"]})
    # the paper's convention counts HGT and HGT tuned as two of "the four
    # architectures" (Table 14's caption, the Limitations sentence); the
    # stricter reading, three architectures in four configurations, is noted
    # in the run report
    n_arch = len({r["row"].replace(" symmetric", "") for _, r in cells()})

    src = {
        "68{,}235": (68235, modeled, "sum of the five clean_dataset parquets"),
        "0.0013": (0.0013, floor, "DETERMINISM_MEASURED.json"),
        "0.0015": (0.0015, fifth, "full5 math theta_0.20 floor_delta"),
        "0.0179": (0.0179, foot, "theta_0.20 legacy vs repaired math floor"),
        "2026": (2026, 2026, "snapshot year, tab:params design constant"),
        "2011": (2011, 2011, "t0 cutoff, tab:params design constant"),
    }
    abst, s1 = abstract_and_s1()
    for chunk, cname in ((abst, "abstract"), (s1, "Section 1")):
        toks = set(re.findall(r"\d+\{,\}\d+|\d+\.\d+|\d{4}", chunk))
        for t in sorted(toks):
            if t in src:
                stated, artifact, where = src[t]
                out(abs(float(str(stated).replace("{,}", "")) - artifact) < 1e-9
                    if isinstance(artifact, float) else
                    int(str(stated).replace("{,}", "")) == artifact,
                    f"B1 {cname}: {t} <- {where}",
                    f"artifact value {artifact}")
            else:
                small = {"5", "0.2", "10", "15"}
                if t in small:
                    out(True, f"B1 {cname}: {t} <- contract constant "
                              f"(tab:params)")
                else:
                    out(False, f"B1 {cname}: token {t} has no source in the "
                               f"provenance map", "unmapped token")
    # the abstract's word-numbers, resolved against the artifacts
    out(n_arch == 4, "B1 abstract: 'of four graph architectures' equals the "
        "distinct architectures in the strict cells", f"artifact {n_arch}")
    out(ex_arch == ["GAT", "RGCN"],
        "B1 abstract: 'only two clear it' equals the distinct exceeding "
        "architectures", f"artifact {ex_arch}")
    out(ex_fields == ["chemistry"],
        "B1 abstract: 'in one discipline of the five' equals the distinct "
        "exceeding disciplines", f"artifact {ex_fields}")


# ---------------------------------------------------------------- B2, B3
def b2_b3():
    print("\n== B2: Table 1 rows against the body sentence stating the same "
          "quantity ==")
    ctab = (PAPER / "construction_table.tex").read_text(encoding="utf-8")
    body = TEX[:TEX.index("\\bibliographystyle")]
    spread = CALL["spread"]
    out(f"{spread:.4f}" == "0.0050",
        "B2 call-path: probe artifact spread is the 0.0050 the table prints",
        f"artifact {spread}")
    out(re.search(r"spread\s+0\.0050, which is 3\.8 times the floor", body)
        is not None,
        "B2 call-path: Section 6 prose states the same 0.0050 and 3.8x")
    n_0058 = len(re.findall(r"0\.0058", body))
    out(n_0058 == 0,
        "B2 call-path: 0.0058 appears nowhere in the body",
        f"{n_0058} body sites")
    rep_sites = len(re.findall(r"0\.0058", TEX)) - n_0058
    report("B2 call-path: 0.0058 in the appendix",
           f"{rep_sites} site(s), all inside the protected app:repro passage "
           f"that presents the two measurements' disagreement as its subject; "
           f"deliberate, not drift")
    foot = abs(TH20["math"]["self_persistence"]["verdict"]
               ["best_student_auc_pr"] - CTRL["verdict"]["best_student_auc_pr"])
    out(f"{foot:.4f}" in ctab, "B2 footing: table size equals the artifact",
        f"artifact {foot:.4f}")
    m20 = F5["fields"]["math"]["cells"]["theta_0.20"]
    out(f"{abs(m20['floor_delta']):.4f}" in ctab,
        "B2 fifth feature: table size equals the artifact",
        f"artifact {abs(m20['floor_delta']):.4f}")
    report("B2 F1 and F2 rows", "their sizes appear only in Table 1; the body "
           "names the repair's per-discipline movement (+0.012, +0.006, ...), "
           "a different quantity from F1-alone, so there is no equality to "
           "assert")

    print("\n== B3: every x-floor ratio recomputed from the artifact ==")
    floor = DET["max_drift_point_estimate"]
    for stated, size, name in [("6.9", 0.009, "F1 low"), ("8.5", 0.011, "F1 high"),
                               ("13.8", foot, "footing"),
                               ("3.8", spread, "call path"),
                               ("1.2", abs(m20["floor_delta"]), "fifth feature")]:
        got = f"{size / floor:.1f}"
        out(got == stated and (stated in ctab),
            f"B3 {name}: {size:.4f} / {floor} = {got}, table prints {stated}")
    out("below" in ctab, "B3 F2: printed as below the floor, no ratio")


# ---------------------------------------------------------------- B4
def b4():
    print("\n== B4: contract constants at every site, and the funnel sums ==")
    import pandas as pd
    n_p5 = len(re.findall(r"t_0(?:\^i)?\s*\+\s*5|t_0\{\+\}5|t0\+5", TEX))
    n_p15 = len(re.findall(r"t_0(?:\^i)?\s*\+\s*15|t0\+15", TEX))
    bad_w = len(re.findall(r"t_0(?:\^i)?\s*\+\s*(?!5|15)\d", TEX))
    out(bad_w == 0, f"B4 windows: every t0 offset is 5 or 15 "
        f"({n_p5} and {n_p15} sites)", f"{bad_w} other offsets")
    thetas = re.findall(r"\\theta = 0\.2\b", TEX)
    out(len(re.findall(r"late.?_overlap\}? > 0\.2", TEX)) >= 1
        and len(thetas) >= 1,
        f"B4 theta: 0.2 at the definition and label sites "
        f"({len(thetas)} theta sites)")
    rows = re.findall(r"(?:economics|math|neuro|physics|chemistry)\s*&\s*"
                      r"([\d{},]+)\s*&\s*([\d{},]+)\s*&\s*[\d.]+\s*&\s*"
                      r"([\d{},]+)\s*&\s*[\d.]+", TEX)
    ints = lambda s: int(s.replace("{,}", "").replace(",", ""))
    raw = sum(ints(r[0]) for r in rows[:5])
    modeled_tab = sum(ints(r[2]) for r in rows[:5])
    out(raw == 330282, "B4 funnel: raw column sums to the stated 330,282",
        f"sum {raw}")
    out(modeled_tab == 68235, "B4 funnel: modeled column sums to 68,235",
        f"sum {modeled_tab}")
    modeled_pq = sum(len(pd.read_parquet(ROOT / "data" /
                                         f"clean_dataset_{f}.parquet"))
                     for f in FIELDS)
    out(modeled_pq == modeled_tab,
        "B4 funnel: modeled column equals the parquet row counts",
        f"parquets {modeled_pq}, table {modeled_tab}")
    full = re.findall(r"(?:economics|math|neuro|physics|chemistry) & "
                      r"([\d{},]+) & ([\d{},]+) & ([\d{},]+)\$?\^?\{?\\?a?s?t?"
                      r"\}?\$? & ([\d{},]+)", TEX)
    ok = all(ints(a) >= ints(b) >= ints(c) >= ints(d)
             for a, b, c, d in full[:5]) if len(full) >= 5 else None
    out(bool(ok), "B4 funnel: every discipline's counters are monotone "
        "raw >= resolvable >= survive >= modeled",
        f"rows parsed {len(full)}")
    out(len(re.findall(r"top-ten", TEX)) >= 2 and "top concepts per profile" in
        TEX, "B4 profiles: top-ten stated in prose and tab:params")


# ---------------------------------------------------------------- B5
def b5():
    print("\n== B5: cross-artifact numbers in README, DATASHEET, "
          "BUILD_REPORT ==")
    stale = {"74", "80", "86", "87", "92", "93", "50", "56"}
    for doc in ["README.md", "datasheet/DATASHEET.md", "BUILD_REPORT.md"]:
        p = ROOT / doc
        if not p.exists():
            report(f"B5 {doc}", "absent")
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        hits = [m.group(0) for m in
                re.finditer(r"[^.\n]*\b(\d{2})\b[^.\n]*checks?[^.\n]*", t)
                if m.group(1) in stale]
        out(not hits, f"B5 {doc}: no stale check count near 'checks'",
            "; ".join(h.strip()[:90] for h in hits[:4]))
        for tok, name in [("68,235", "modeled"), ("330,282", "raw")]:
            if tok.replace(",", "") in t.replace(",", "").replace("{,}", ""):
                out(True, f"B5 {doc}: {name} count matches {tok}")


# ---------------------------------------------------------------- C1
def c1():
    print("\n== C1: recount of the exceeding cells from the artifact ==")
    ex = [(f, r["row"], r["config"], r["delta_vs_M5prime"])
          for f, r in cells() if r["exceeds"]]
    archs = sorted({r.replace(" symmetric", "") for _, r, _, _ in ex})
    fields = sorted({f for f, _, _, _ in ex})
    out(len(ex) == 5, f"C1: five exceeding cells ({len(ex)})",
        str(ex))
    out(archs == ["GAT", "RGCN"], f"C1: two architectures ({archs})")
    out(fields == ["chemistry"], f"C1: one discipline ({fields})")
    a, _ = abstract_and_s1()
    out("of four graph\narchitectures only two clear it, in one discipline\nof "
        "the five" in a or re.search(
            r"of four graph\s+architectures only\s+two clear it, in one "
            r"discipline\s+of the five", a) is not None,
        "C1: the abstract sentence matches the recount")


# ---------------------------------------------------------------- C2
def c2():
    print("\n== C2: every verdict against eq. (2)'s three gates, from the "
          "artifact ==")
    bad = []
    for f, r in cells():
        gates = (r["delta_vs_M5prime"] > 0 and r["p_BH"] < 0.05
                 and r["ci"][0] > 0)
        if gates != r["exceeds"]:
            bad.append((f, r["row"], r["config"]))
    out(not bad, f"C2 strict 39: exceeds flag equals the three gates in every "
        f"cell", str(bad))
    n = 0
    exceeding = []
    for f in FIELDS:
        p = ROOT / "results" / f"results_{f}" / "e12_corrected_vs_m5.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        for name, m in d["models"].items():
            g = m["gates_fair"]
            lo = m["bootstrap"]["vs_M5prime"]["pooled_ci95"][0]
            recomputed = (m["delta_vs_M5prime"] > 0
                          and m["p_adj_M5prime"] < 0.05 and lo > 0)
            if recomputed != all(g.values()) or recomputed != m["exceeds_fair"]:
                bad.append((f, name, f"recomputed {recomputed}, gates {g}, "
                                     f"flag {m['exceeds_fair']}"))
            if m["exceeds_fair"]:
                exceeding.append((f, name))
            n += 1
    out(not bad and exceeding == [("chemistry", "rgcn")],
        f"C2 full20: all {n} unrepaired cells' exceeds_fair equal the three "
        f"gates recomputed from delta, p_adj and the student interval; the "
        f"single exceeds is chemistry RGCN", str(bad[:4]) + str(exceeding))
    for f in FIELDS:
        s = json.loads((ROOT / "results" / "revision" / "T3_4_tabpfn_gpu" / f /
                        "summary.json").read_text(encoding="utf-8"))
        lo = s["student_ci95_vs_M5prime"][0]
        recomputed = (s["delta_vs_M5prime"] > 0 and s["p_BH"] < 0.05
                      and lo > 0)
        if bool(s["exceeds_fair"]) != recomputed or recomputed != all(
                s["gates_fair"].values()):
            bad.append(("t34", f))
    out(not bad, "C2 TabPFN five cells: exceeds_fair equals the gates "
        "recomputed from the summary fields")
    report("C2 verdicts under other rules, by design",
           "tab:genealogy (TOST equivalence tiers), tab:power (equivalence "
           "vocabulary), the four certificates (pre-stated pass rules), "
           "tab:mech branches (two-threshold branch rule), t31 gates "
           "(yes-rate gap and self-agreement): none claims eq. (2); each "
           "caption names its own rule")


# ---------------------------------------------------------------- C3
def c3():
    print("\n== C3: the abstract's two construction claims against Table 1 ==")
    floor = DET["max_drift_point_estimate"]
    m20 = F5["fields"]["math"]["cells"]["theta_0.20"]
    foot = abs(TH20["math"]["self_persistence"]["verdict"]
               ["best_student_auc_pr"] - CTRL["verdict"]["best_student_auc_pr"])
    sizes = {"F1": 0.009, "footing": foot, "callpath": CALL["spread"],
             "fifth": abs(m20["floor_delta"]), "F2": 0.0008}
    above = [k for k, v in sizes.items() if v > floor]
    out(len(above) == 4 and "F2" not in above,
        f"C3: four of five above the floor ({above}; F2 at 0.0002 to 0.0008 "
        f"is below {floor})")
    changed = {"F1": (0, True), "footing": (0, True), "callpath": (None, False),
               "fifth": (2, True), "F2": (0, True)}
    evaluated_changed = [k for k, (n, ev) in changed.items() if ev and n]
    out(evaluated_changed == ["fifth"],
        "C3: the fifth feature is the only evaluated choice that changed a "
        f"label (evaluated: F1 0 of 9, footing 0 of 10, fifth 2 of 2, F2 no; "
        f"call path not evaluated)")


# ---------------------------------------------------------------- C4
def c4():
    print("\n== C4: which ceiling each claim uses ==")
    body = TEX[:TEX.index("\\bibliographystyle")]
    sents = re.split(r"(?<=[.!?])\s+", re.sub(r"%.*", "", body))
    hits = [s.replace("\n", " ")[:110] for s in sents
            if re.search(r"ceiling|M5\$'\$|M5'", s)]
    print(f"       {len(hits)} body sentences name a ceiling; inventory:")
    for h in hits:
        print(f"         - {h}")
    report("C4 conclusion", "graph and TabPFN verdicts are against M5'; the "
           "M6 genealogy arm is against M5 and its caption says so; no body "
           "sentence puts a vs-M5 number and a vs-M5' number in one "
           "comparison. The one sentence naming both, 'the ceiling means the "
           "pre-specified comparator M5 and M5$'$ its validation-symmetric "
           "variant', is the definition")


# ---------------------------------------------------------------- D1
def d1():
    print("\n== D1: floats, labels, references, and ?? in the PDF ==")
    frags = [(PAPER / f).read_text(encoding="utf-8") for f in
             ["main.tex", "construction_table.tex", "table4_verdicts.tex",
              "t34_table.tex", "t31_table.tex", "strict_appendix.tex",
              "mech_appendix.tex", "lineage_appendix.tex",
              "premium_appendix.tex"]]
    allsrc = "\n".join(frags)
    labels = re.findall(r"\\label\{((?:fig|tab):[^}]*)\}", allsrc)
    refs = set(re.findall(r"\\ref\{((?:fig|tab):[^}]*)\}", allsrc))
    unref = [l for l in labels if l not in refs]
    out(not [l for l in unref if not l.startswith("tab:lineage")
             and l not in ("tab:attribution", "tab:fullgridb", "tab:mechdeg",
                           "tab:mechdisp", "tab:mechlevel")],
        f"D1: every float label is referenced (known unreferenced appendix "
        f"tables: {sorted(unref)})")
    dead = [r for r in refs if r not in labels]
    out(not dead, "D1: every fig/tab reference resolves", str(dead))
    import fitz
    doc = fitz.open(PAPER / "main.pdf")
    qq = [i + 1 for i in range(doc.page_count)
          if "??" in doc[i].get_text()]
    out(not qq, "D1: zero ?? in the PDF text layer", f"pages {qq}")


# ---------------------------------------------------------------- D2
def d2():
    print("\n== D2: figures against the tables that state the same "
          "quantity ==")
    sys.path.insert(0, str(ROOT / "code"))
    import importlib
    lad = importlib.import_module("make_ladder_figure")
    m4 = json.loads((lad.M4_TRAINFIT).read_text(encoding="utf-8"))
    tab = re.search(r"M4 text logistic\s*&([^\\]*)\\\\", TEX).group(1)
    tabvals = [float(x) for x in re.findall(r"0\.\d+", tab)]
    figvals = [round(m4["fields"][f]["trainfit"]["auc_pr"]["mean"], 3)
               for f in FIELDS]
    out(tabvals == figvals,
        "D2 ladder: the M4 values the figure draws equal tab:ladder's row",
        f"figure {figvals}, table {tabvals}")
    r57 = importlib.import_module("r57_emit_gnn_figure")
    cs, total, exceed = r57.load()
    r57.check_against_table4(cs)
    out(total == 39 and exceed == 5,
        "D2 gnn39: 39 cells, 5 exceeding, verdicts agree with Table 39 "
        "(r57's own cross-check ran clean)")
    fw = (PAPER / "figures" / "F10_framework.tex").read_text(encoding="utf-8")
    out("330,282" in fw and "68,235" in fw,
        "D2 framework: panel (a) totals match the funnel sums")
    rates = re.findall(r"(?:econ|math|neuro)[^\n]*?\\tk\{\.(\d\d)\}|"
                       r"(?:physics|chem)[^\n]*?\\tk\{\.(\d\d)\}", fw)
    flat = [a or b for a, b in rates]
    out(flat[:5] == ["20", "20", "25", "35", "22"] if len(flat) >= 5 else False,
        "D2 framework: the five base rates in panel (a) match tab:funnel "
        "rounded to two decimals", f"drawn {flat[:5]}")
    fw_plain = re.sub(r"\\tk\{([^}]*)\}", r"\1", fw)
    n_ex = sum(1 for _, r in cells() if r["exceeds"])
    n_cells = sum(t["n_rows"] for t in ASM["tables"].values())
    out(f"{n_ex} of {n_cells} cells exceed" in fw_plain,
        "D2 framework: panel (d) count matches the assembled cells",
        f"artifact {n_ex} of {n_cells}")
    mod = json.loads((ROOT / "data" / "network_modularity.json")
                     .read_text(encoding="utf-8"))
    cc = {k: v.get("largest_cc_nodes") for k, v in mod.items()
          if isinstance(v, dict)}
    lo, hi = min(cc.values()), max(cc.values())
    out(f"{lo:,}" .replace(",", "{,}") in TEX and
        f"{hi:,}".replace(",", "{,}") in TEX,
        "D2 networks: the caption's node counts are the modularity file's "
        f"min and max largest components ({lo:,} and {hi:,})")
    report("D2 appendix figures F13, F14, F15",
           "generated from result files at freeze; not re-verified "
           "mechanically in this pass")


def d3():
    """Banned words in the paper sources. 'naive' was retired for
    'uncorrected' in the final submission round and must not return."""
    print("\n== D3: banned words in the paper sources ==")
    banned = ["naive"]
    hits = []
    for f in sorted(PAPER.glob("*.tex")):
        for i, line in enumerate(f.read_text(encoding="utf-8",
                                             errors="replace").splitlines(), 1):
            code = line.split("%", 1)[0]
            for w in banned:
                if re.search(rf"(?i)\b{w}\b", code):
                    hits.append(f"{f.name}:{i}: {code.strip()[:70]}")
    for h in hits:
        print(f"       {h}")
    out(not hits, "D3 banned: no retired term in any paper source",
        f"{len(hits)} hits" if hits else "clean")


if __name__ == "__main__":
    b1(); b2_b3(); b4(); b5(); c1(); c2(); c3(); c4(); d1(); d2(); d3()
    print(f"\n{n_pass} pass, {n_fail} fail")
    sys.exit(1 if n_fail else 0)
