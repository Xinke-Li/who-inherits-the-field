#!/usr/bin/env python3
"""Declare what the next Zenodo version must carry, with sizes and hashes.

This script prepares a deposit. It cannot perform one: it has no network code
and writes a single JSON file. Uploading is a hand step in PUBLISH_STEPS.md.

WHY A DECLARATION AND NOT A CHECK OF THE CURRENT ARCHIVE. r43 verifies the
deposited zenodo_archive.zip against the tree, which catches drift in what was
already deposited and is silent about what was never deposited at all. The
123 MB concept-event supplement is exactly that case: a shipped assertion reads
it, it is far too large for git, and the archive predates it, so every check
passes while a reader cannot re-run T2.11.

THE RULE, applied to every file in data/supplement/. A supplement artifact is
REQUIRED in the archive when something in the release reads it and it is too
large for git. It is a CANDIDATE when it is an input a reader might want but no
released claim depends on. It is NEITHER when git already carries it, or when it
is a regenerable working cache.

Output: results/revision/DEPOSIT_MANIFEST.json

  python code/r53_deposit_manifest.py
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "data" / "supplement"
ARCHIVE = ROOT / "zenodo_archive.zip"
OUT = ROOT / "results" / "revision" / "DEPOSIT_MANIFEST.json"

REQUIRED = {
    "concept_events_*.parquet": (
        "reproduction/verify_time_contract.py rebuilds early concepts from "
        "these and is what makes T2.11's exact t0+5 assertion checkable; the "
        "harness reads the recorded verdict either way, but without the tables "
        "a reader can only read that verdict, never re-derive it. 120 MB is "
        "past what git should carry"),
}
CANDIDATE = {
    "titles_*.parquet": (
        "input to T3.1's adjudication sample. The 2,200 rendered items and the "
        "judge's raw answers are both tracked, so a reader can recompute every "
        "T3.1 number without these; they are what a reader redrawing the sample "
        "from scratch would need"),
}
EXCLUDED = {
    "abstracts_*.parquet": (
        "193.5 MB, and excluded by decision rather than by size. Four of the "
        "five disciplines have them: chemistry's merge completed titles-only "
        "after the full merge exhausted the machine. Depositing four fifths of "
        "an input makes the first question about the release the whereabouts of "
        "chemistry, and the input in question feeds a text rung this paper does "
        "not run, so shipping it would advertise an absent baseline rather than "
        "support a present one. Nothing reported reads them"),
    "results/revision/T2_6_funnel_neuro/works_cache/": (
        "287 MB of T2.6 fetch input, superseded by a derived file 130 times "
        "smaller. funnel_neuro_years.parquet, 2.2 MB and tracked in git, "
        "reproduces all eight reconstructed funnel counters through "
        "r49_funnel_neuro_counters.py --stage verify"),
}
NEITHER = {
    "early_concentration_*.parquet": "git carries it, 824 KB",
    "lineage_*.parquet": "git carries it, 1.8 MB",
    "funnel_neuro_years.parquet": "git carries it, 2.2 MB",
    "*.json": "git carries every supplement manifest",
    "_ta_cache/": ("regenerable working cache of the B1 fetch, 2.88 GB; the "
                   "per-field outputs are what any consumer reads"),
}

# Every pattern must be classified exactly once, or a file added later drifts
# into whichever bucket happens to match it first.
_ALL = [REQUIRED, CANDIDATE, EXCLUDED, NEITHER]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def entries_for(pattern):
    """Patterns are relative to data/supplement/, except the one excluded tree
    that lives under results/revision/ and is named with its own prefix."""
    if pattern.startswith("results/"):
        base = ROOT / pattern.rstrip("/")
        return sorted(q for q in base.rglob("*") if q.is_file())
    return sorted(SUP.glob(pattern))


def describe(paths):
    out = []
    for p in paths:
        out.append({"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": p.stat().st_size,
                    "mb": round(p.stat().st_size / 1e6, 2),
                    "sha256": sha256(p)})
    return out


def main():
    in_archive = set()
    if ARCHIVE.exists():
        with zipfile.ZipFile(ARCHIVE) as z:
            in_archive = {i.filename for i in z.infolist() if not i.is_dir()}

    man = {"purpose": "what the next Zenodo version must carry, and why",
           "this_script_cannot_upload": True,
           "current_archive": {
               "file": ARCHIVE.name if ARCHIVE.exists() else None,
               "n_entries": len(in_archive)},
           "required": [], "candidate": [],
           "excluded_by_decision": [],
           "excluded_because_git_carries_it": NEITHER}

    for tier, spec in (("required", REQUIRED), ("candidate", CANDIDATE),
                       ("excluded_by_decision", EXCLUDED)):
        for pattern, why in spec.items():
            files = describe(entries_for(pattern))
            for f in files:
                f["already_in_archive"] = f["path"] in in_archive
            man[tier].append({
                "pattern": pattern, "why": why, "files": files,
                "n_files": len(files),
                "total_mb": round(sum(f["mb"] for f in files), 2),
                "pending": [f["path"] for f in files
                            if not f["already_in_archive"]]})

    req_pending = [p for g in man["required"] for p in g["pending"]]
    man["pending_required_count"] = len(req_pending)
    man["pending_required"] = req_pending
    man["harness_consequence"] = (
        "one check moves from SKIP to PASS once the required files are "
        "deposited and unpacked, 'supplement: every concept event table "
        "exists'. The other two concept-event checks and the two time-contract "
        "checks read tracked JSON and already run. The larger consequence is "
        "not the count: verify_time_contract.py cannot execute at all without "
        "these tables, so T2.11 is readable but not re-runnable by a third "
        "party until they are deposited")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(man, indent=2), encoding="utf-8")
    for g in man["required"]:
        print(f"[r53] REQUIRED  {g['pattern']:34s} {g['n_files']} files, "
              f"{g['total_mb']:7.1f} MB, {len(g['pending'])} pending")
    for g in man["candidate"]:
        print(f"[r53] candidate {g['pattern']:34s} {g['n_files']} files, "
              f"{g['total_mb']:7.1f} MB, {len(g['pending'])} pending")
    for g in man["excluded_by_decision"]:
        print(f"[r53] EXCLUDED  {g['pattern']:34s} {g['n_files']} files, "
              f"{g['total_mb']:7.1f} MB, not deposited by decision")
    print(f"[r53] -> {OUT}")


if __name__ == "__main__":
    main()
