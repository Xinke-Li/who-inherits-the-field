#!/usr/bin/env python3
"""Prepare the single release. Never perform it.

This script has no path to the network and cannot acquire one at runtime. Two
guarantees, both enforced rather than asserted in a comment:

  1  socket.socket, socket.create_connection and socket.getaddrinfo are
     replaced at import with a function that raises, so any library that tried
     to open a connection dies rather than succeeding quietly.
  2  every git call goes through git(), which refuses any subcommand outside a
     read-only allowlist. push, remote add, fetch and pull are not in it, so
     this file cannot be edited into an uploader by accident.

The release itself is a deliberate act the author performs by hand, once, after
the paper is final, following PUBLISH_STEPS.md. What this does is tell you
whether that act would be safe, and hand you the manifest to check against
afterwards.

TWO TARGETS, NOT ONE. PUBLISH_STEPS.md describes two artifacts with different
contents, and conflating them was this script's first bug:

  GitHub main   master's tracked files minus paper/, rebuilt as a single orphan
                commit. Roughly 1,800 files.
  Zenodo        zenodo_archive.zip, a curated set the record already carries.
                Its membership is read from the existing archive rather than
                recomputed here, because PUBLISH_STEPS.md requires the archive
                stay byte-identical across versions so the hashes printed in
                the paper keep verifying. This script therefore verifies the
                archive; it never proposes one.

UNTRACKED FILES ARE NOT ALL EQUAL. A working document sitting in the tree is
not the same failure as an artifact the paper or the assertion harness reads
being absent from git. Check 1 separates them: a file is blocking only if its
path is named in paper/main.tex, reproduction/reproduce_assertions.py or
data/SHA256SUMS. Everything else is listed and does not stop the run.

Five pre-flight checks, each refusing with its own exit code:

  2  the GitHub tree is not what the recipe says it is, an artifact something
     reads is untracked, or the Zenodo archive does not verify
  3  an automated-tool attribution appears in a shipped file or in the git
     identity of the commits that carry them
  4  the release is not solely authored by Xinke Li
  5  a line of data/SHA256SUMS does not verify
  6  reproduction/reproduce_assertions.py does not exit 0

  python code/r43_prepare_release.py
  python code/r43_prepare_release.py --skip-assertions   # drop the slow check
  python code/r43_prepare_release.py --diagnose          # every check, no manifest

--diagnose runs all five and reports all of them, so the author sees the whole
picture in one pass instead of fixing one blocker and discovering the next. It
never writes a manifest and never exits 0 with a failure outstanding: a
diagnosis is not a clearance.

Output: results/revision/RELEASE_MANIFEST.json
"""
import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class NoNetwork(RuntimeError):
    pass


def _blocked(*a, **k):
    raise NoNetwork(
        "r43 has no network path by construction. If you are seeing this, "
        "something in this process tried to open a connection; that is a bug, "
        "not a configuration problem.")


socket.socket = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision" / "RELEASE_MANIFEST.json"
EXCLUDED_PREFIX = "paper/"
# The public tree withholds these tracked working records besides paper/;
# build_release keeps this list and r43 mirrors it.
EXCLUDED_FILES = {
    "PUBLISH_STEPS.md",
    "results/revision/HANDOVER.md",
    "results/revision/SUBMISSION_CHECKLIST.md",
    "results/revision/T5_3_response_to_reviewers.md",
    "results/revision/T4_4_anonymization_checklist.md",
    "results/revision/T1_framing_checklist.md",
    "results/revision/T0_audit_index.md",
    "results/revision/T0_4_history_audit.md",
    "results/revision/COMPRESSION_PLAN.md",
    "results/revision/PAGE_LOG.md",
    "results/revision/FORWARD_REFERENCE_DEBT.md",
    "results/revision/README.md",
    "results/revision/T2_11_time_contract/b2.log",
    "results/revision/T2_6_funnel_neuro/count.log",
    "results/revision/T1_tex.diff",
}
# The public root carries exactly these thirteen entries; any other root entry
# or any excluded-pattern basename in the shipped set is a hard failure.
ROOT_MANIFEST = sorted([".gitattributes", ".gitignore", "BUILD_REPORT.md",
                        "CITATION.cff", "LEAKAGE_AUDIT_e12.md", "LICENSE",
                        "README.md", "code", "colab", "data", "datasheet",
                        "reproduction", "results"])
EXCLUDE_BASENAME_PREFIXES = ("openreview_", "zenodo_", "github_about",
                             "CHANGES.md", "style_guide", "avoiding",
                             "HANDOVER", "PUBLISH_STEPS")
ARCHIVE = ROOT / "zenodo_archive.zip"
# Files whose contents name the artifacts that must exist. An untracked path
# named in one of these is blocking; an untracked path named in none of them is
# a working document.
READERS = ["paper/main.tex",
           "reproduction/reproduce_assertions.py",
           "data/SHA256SUMS"]
AUTHOR_NAME = "Xinke Li"
AUTHOR_EMAIL = "lixinke@uchicago.edu"
ORCID = "0009-0001-0403-3606"

GIT_READONLY = {"ls-files", "log", "status", "rev-parse", "config", "remote",
                "for-each-ref", "cat-file", "hash-object", "check-ignore"}

# Text extensions only. The genealogy tables carry researcher names, among them
# Claude and Jean-Claude, so a scan over their bytes would be meaningless.
TEXT_SUFFIX = {".py", ".md", ".txt", ".tex", ".cff", ".json", ".yml", ".yaml",
               ".sh", ".ipynb", ".bib", ".cls", ".sty", ".cfg", ".toml", "",
               ".gitignore", ".gitattributes", ".robustness", ".caches"}

# Patterns chosen so the two legitimate appearances do not fire: "Claude" and
# "Jean-Claude" are researcher names in the genealogy, and "anthropic" is a
# substring of "philanthropic". Neither matches below.
ATTRIB = [
    (r"Co-[Aa]uthored-[Bb]y", "commit trailer"),
    (r"\bClaude Code\b", "tool name"),
    (r"noreply@anthropic\.com", "tool email"),
    (r"(?<!phil)(?<!Phil)\bAnthropic\b", "vendor name"),
    (r"\bGitHub Copilot\b", "tool name"),
    (r"\bChatGPT\b", "tool name"),
    (r"[Gg]enerated with .{0,40}(Claude|Copilot|GPT)", "generation notice"),
]

# Narrow, visible exceptions. Every one is printed as a NOTE rather than
# suppressed, so an auditor sees what was allowed and why. A file earns a place
# here only by naming a pattern in order to assert its absence.
ALLOWED = {
    ("results/revision/T0_4_history_audit.md", "commit trailer"):
        "the audit quotes the trailer inside the grep that looks for it and "
        "inside the sentence recording that no commit carries one",
    ("code/r42_run_judge.py", "vendor name"):
        "the file calls a vendor API and must name the SDK it imports, the "
        "same way another file names pandas; the vendor is a dependency here "
        "and an instrument in app:aitools, not an author or a contributor",
    ("results/revision/SUBMISSION_CHECKLIST.md", "tool name"):
        "the section 3 block is the verbatim text for the submission system's "
        "generative-AI disclosure field, which cannot disclose a tool without "
        "naming it; the same disclosure appears in Methods and app:aitools, "
        "and it states that no submission text was generated and that the "
        "author verified every number",
}


def git(*args, check=True):
    if not args or args[0] not in GIT_READONLY:
        raise SystemExit(
            f"r43: refusing git {args[0] if args else '(nothing)'}. Only "
            f"{sorted(GIT_READONLY)} are permitted here; this script does not "
            f"publish.")
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise SystemExit(f"r43: git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_text(p):
    return p.suffix.lower() in TEXT_SUFFIX or p.name.startswith(".git")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-assertions", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    a = ap.parse_args()
    failed = []

    def stop(code, msg):
        """Refuse. Under --diagnose, record and carry on to the next check."""
        print(msg)
        failed.append(code)
        return None if a.diagnose else code

    print("r43: preparing a release description. This script cannot publish.")
    print(f"  network: sockets blocked at import ({NoNetwork.__name__})")
    print(f"  git:     read-only allowlist, {len(GIT_READONLY)} subcommands")
    print()

    # ---- the shipped tree ----
    tracked = [l for l in git("ls-files").splitlines() if l.strip()]
    shipped = sorted(p for p in tracked if not p.startswith(EXCLUDED_PREFIX)
                     and p not in EXCLUDED_FILES)
    withheld = sorted(p for p in tracked if p not in shipped)
    print(f"[1/5] shipped tree: {len(shipped)} files "
          f"({len(tracked)} tracked minus {len(withheld)} withheld: "
          f"{EXCLUDED_PREFIX} plus {len(EXCLUDED_FILES)} working records)")

    # ---- the root manifest gate ----
    roots = sorted({p.split("/", 1)[0] for p in shipped})
    bad_pattern = [p for p in shipped if any(
        p.rsplit("/", 1)[-1].startswith(x) for x in EXCLUDE_BASENAME_PREFIXES)
        or p.endswith(".figblock")]
    if roots != ROOT_MANIFEST or bad_pattern:
        for r in roots:
            if r not in ROOT_MANIFEST:
                print(f"  ROOT ENTRY OUTSIDE THE MANIFEST: {r}")
        for r in ROOT_MANIFEST:
            if r not in roots:
                print(f"  MANIFEST ENTRY ABSENT: {r}")
        for p in bad_pattern[:10]:
            print(f"  EXCLUDED-PATTERN FILE IN SHIPPED SET: {p}")
        rc = stop(1, "STOPPING: the shipped root does not equal the "
                     "thirteen-entry manifest.")
        if rc:
            return rc

    missing = [p for p in shipped if not (ROOT / p).exists()]
    if missing:
        for p in missing[:10]:
            print(f"  MISSING FROM DISK: {p}")
        rc = stop(2, "STOPPING: a tracked file the release would carry is not "
                     "present.")
        if rc:
            return rc

    untracked = [l[3:].strip().strip('"') for l in
                 git("status", "--porcelain", "--untracked-files=all"
                     ).splitlines() if l.startswith("??")]
    # git status never lists an ignored file, so a blanket rule like *.parquet
    # hides an artifact from this check completely. That is the same failure as
    # the globbed-directory case fixed below, arriving by a different route: the
    # file a reader reads is absent from the clone and the check reports clean.
    # Fifteen supplement parquets, among them the five the harness opens by
    # name, sat in exactly that blind spot. Ignored paths are therefore
    # candidates too, and are reported separately because the remedy differs:
    # an untracked file needs git add, an ignored one needs git add -f or a
    # negation in .gitignore.
    ignored = [l.strip().strip('"') for l in
               git("ls-files", "--others", "--ignored", "--exclude-standard"
                   ).splitlines() if l.strip()]
    inside = [p for p in untracked if not p.startswith(EXCLUDED_PREFIX)]
    inside_ignored = [p for p in ignored if not p.startswith(EXCLUDED_PREFIX)]

    # An untracked path is blocking only if something reads it.
    named = ""
    for r in READERS:
        f = ROOT / r
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8", errors="ignore")
        if r.endswith(".tex"):
            # A path named in a LaTeX comment is a pointer for a human, not a
            # file the build opens. Strip comments before deciding.
            t = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in t.splitlines())
        named += t
    # Matching the full relative path alone is not enough, and missing that is
    # how 640 files of T2_1_final, the tree behind Tables 12b and 13b, passed
    # this check while a public clone could not run nine assertions that read
    # them. Two reasons it failed: git lists untracked FILES, and a reader
    # names a DIRECTORY and globs under it; and the harness builds that
    # directory with os.path.join over fragments, so no literal path exists to
    # match at all.
    #
    # So also treat a file as read when one of its ancestor directories is
    # named. A bare basename would flag everything, so an ancestor counts only
    # when its name is distinctive: at least eight characters and carrying a
    # digit or an underscore. That admits T2_1_final and T2_2b_lineage_contract
    # and excludes plain words like chemistry.
    def distinctive(name):
        return len(name) >= 8 and any(c.isdigit() or c == "_" for c in name)

    # A path under a *_out directory is a raw Colab download kept as staging
    # evidence, not the canonical tree a reader runs against. The canonical
    # copies are T2_1_final and T2_2b_lineage_contract, and both are tracked.
    # Without this the ancestor rule matches 939 staging files through the
    # T2_1_strict_contract segment they share with the tree that is read.
    def staging(parts):
        return any(seg.endswith("_out") for seg in parts[:-1])

    def is_read(path):
        stem = path.rstrip("/")
        if stem in named:
            return True
        parts = stem.split("/")
        if staging(parts):
            return False
        for i in range(len(parts) - 1, 0, -1):
            anc = parts[i - 1]
            if distinctive(anc) and anc in named:
                return True
        return False

    # The ancestor rule is right for untracked paths and wrong for ignored
    # ones. What .gitignore holds is dominated by logs, zips and per-seed
    # dumps, and a distinctive ancestor directory sweeps all of them up: the
    # first run of this check flagged option3.log, two staging zips, the
    # deposited archive itself, and two e9a per-seed jsonl files, none of them
    # named anywhere. So an ignored path counts as read when a reader names it
    # literally, or when it is a data table and an ancestor is named. That
    # still catches early_concentration_<field>.parquet, which no literal path
    # matches because the paper writes the field as a placeholder and the
    # harness builds the name with an f-string.
    DATA_SUFFIX = (".parquet", ".csv")
    # One file is named by the readers and correctly absent from git: the
    # deposited archive. The paper names it so a reader can fetch it from the
    # deposit, which is the whole point, and check 4 below verifies it against
    # the tree entry by entry. Exempting it by name is honest; loosening the
    # rule to let it through would also let the next real miss through.
    DEPOSIT_ONLY = {"zenodo_archive.zip"}

    def is_read_ignored(path):
        if path.rstrip("/") in DEPOSIT_ONLY:
            return False
        if path.rstrip("/") in named:
            return True
        return path.endswith(DATA_SUFFIX) and is_read(path)

    blocking, working = [], []
    for p in inside:
        (blocking if is_read(p) else working).append(p)
    blocked_by_ignore = [p for p in inside_ignored if is_read_ignored(p)]

    for p in blocking:
        print(f"  UNTRACKED AND READ: {p}")
    for p in blocked_by_ignore:
        print(f"  IGNORED AND READ: {p}")
    for p in working[:8]:
        print(f"  untracked working document: {p}")
    if len(working) > 8:
        print(f"  ... and {len(working) - 8} more working documents")
    if blocking or blocked_by_ignore:
        rc = stop(2, f"STOPPING: {len(blocking)} untracked and "
                     f"{len(blocked_by_ignore)} ignored path(s) are named in "
                     f"{', '.join(READERS)}. A clone would not contain them, so "
                     f"the paper or the assertion harness would read a file "
                     f"that is not there.")
        if rc:
            return rc
    else:
        print(f"      every artifact the paper or the harness reads is "
              f"tracked, including against .gitignore; {len(working)} untracked "
              f"working document(s) and {len(inside_ignored)} ignored path(s) "
              f"remain, none of them read")

    # ---- the Zenodo archive, verified against itself, never redefined ----
    if not ARCHIVE.exists():
        rc = stop(2, f"STOPPING: {ARCHIVE.name} is absent. PUBLISH_STEPS.md "
                     f"requires it stay byte-identical across versions, so it "
                     f"is verified here, never rebuilt.")
        if rc:
            return rc
        arc = None
    else:
        import zipfile
        with zipfile.ZipFile(ARCHIVE) as z:
            entries = [(i.filename, i.file_size, i.CRC) for i in z.infolist()
                       if not i.is_dir()]
        # Drift inside data/ is blocking: those are the frozen tables whose
        # SHA-256 the paper prints, and PUBLISH_STEPS.md requires their bytes
        # stay fixed across versions. Drift outside data/ is documentation and
        # tooling, which moves between versions by design; it is reported so
        # the author knows what the deposited copy is behind on.
        drift, behind = [], []
        for name, size, _ in entries:
            f = ROOT / name
            if not f.exists():
                (drift if name.startswith("data/") else behind).append(
                    f"{name}: in the archive, absent from the tree")
            elif f.stat().st_size != size:
                msg = (f"{name}: tree {f.stat().st_size} bytes, "
                       f"archive {size}")
                (drift if name.startswith("data/") else behind).append(msg)
        # Drift is only half of it. This check compares the archive to what it
        # already carries and is silent about an artifact that was never
        # deposited, which is how a 123 MB supplement that a shipped assertion
        # reads sat outside the archive with every check green. The deposit
        # manifest declares what the next version must carry; verify that each
        # required file still exists and still hashes to what was declared.
        dep = ROOT / "results" / "revision" / "DEPOSIT_MANIFEST.json"
        pending, stale = [], []
        if dep.exists():
            dm = json.loads(dep.read_text(encoding="utf-8"))
            for group in dm.get("required", []):
                for f in group["files"]:
                    q = ROOT / f["path"]
                    if not q.exists():
                        stale.append(f"{f['path']}: declared, absent from tree")
                    elif sha256(q) != f["sha256"]:
                        stale.append(f"{f['path']}: hash moved since declared")
                    elif not f["already_in_archive"]:
                        pending.append(f["path"])
            # The exclusions are a decision, so make them checkable rather
            # than documentary: a file ruled out of the deposit must not turn
            # up in the archive, or the ruling has been silently reversed.
            wrongly = []
            names = {n for n, _, _ in entries}
            for group in dm.get("excluded_by_decision", []):
                for f in group["files"]:
                    if f["path"] in names:
                        wrongly.append(f"{f['path']}: excluded by decision, "
                                       f"present in the archive")
            stale.extend(wrongly)
            for p in stale:
                print(f"  DEPOSIT MANIFEST STALE: {p}")
            if pending:
                print(f"      {len(pending)} file(s) declared required for the "
                      f"next version and not yet deposited; "
                      f"{dm['pending_required_count']} at declaration time")
        arc = {"file": ARCHIVE.name,
               "zip_bytes": ARCHIVE.stat().st_size,
               "deposit_pending_required": pending,
               "deposit_manifest_stale": stale,
               "n_entries": len(entries),
               "uncompressed_bytes": sum(s for _, s, _ in entries),
               "sha256": sha256(ARCHIVE),
               "entries": [n for n, _, _ in entries],
               "frozen_table_drift": drift,
               "deposited_copy_behind": behind}
        print(f"      Zenodo archive: {len(entries)} entries, "
              f"{ARCHIVE.stat().st_size / 1e6:.1f} MB on disk, "
              f"{sum(s for _, s, _ in entries) / 1e6:.1f} MB uncompressed; "
              f"every data/ entry matches the tree")
        for b in behind:
            print(f"  DEPOSITED COPY BEHIND: {b}")
        if behind:
            print("      the frozen tables are unchanged, so the hashes the "
                  "paper prints still verify against the published record; "
                  "the entries above are documentation and tooling that moved "
                  "after the deposit")
        for d in drift[:6]:
            print("  FROZEN TABLE DRIFT:", d)
        if drift:
            rc = stop(2, f"STOPPING: {len(drift)} frozen table(s) no longer "
                         f"match the published archive. The paper's printed "
                         f"hashes are read against those bytes.")
            if rc:
                return rc

    # ---- attribution ----
    hits, notes, scanned, skipped = [], [], 0, 0
    for rel in shipped:
        p = ROOT / rel
        if not is_text(p):
            skipped += 1
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            skipped += 1
            continue
        scanned += 1
        for pat, what in ATTRIB:
            for m in re.finditer(pat, txt):
                ln = txt.count("\n", 0, m.start()) + 1
                why = ALLOWED.get((rel.replace("\\", "/"), what))
                if why:
                    notes.append(f"{rel}:{ln}  {m.group(0)!r}  allowed: {why}")
                else:
                    hits.append(f"{rel}:{ln}  {what}  {m.group(0)!r}")
    log = git("log", "--all", "--format=%an|%ae|%cn|%ce|%s|%b")
    for pat, what in ATTRIB:
        for m in re.finditer(pat, log):
            hits.append(f"(git log)  {what}  {m.group(0)!r}")
    print(f"[2/5] attribution scan: {scanned} text files, {skipped} binary or "
          f"unreadable, plus every commit message and identity")
    for n in notes:
        print("  NOTE:", n)
    if hits:
        for h in hits[:20]:
            print("  ATTRIBUTION:", h)
        rc = stop(3, f"STOPPING: {len(hits)} automated-tool attributions in "
                     f"material that would ship.")
        if rc:
            return rc
    else:
        print("      clean")

    # ---- sole author ----
    problems = []
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    def unquote(s):
        return s.strip().strip('"').strip("'").strip()

    names = [unquote(x) for x in re.findall(r"family-names:\s*(.+)", cff)]
    givens = [unquote(x) for x in re.findall(r"given-names:\s*(.+)", cff)]
    people = [f"{g} {n}" for g, n in zip(givens, names)]
    # the dataset block and the preferred-citation block each name the author,
    # so the same person may appear more than once; a second PERSON still fails
    if sorted(set(people)) != [AUTHOR_NAME]:
        problems.append(f"CITATION.cff lists {people}")
    if ORCID not in cff:
        problems.append(f"CITATION.cff does not carry ORCID {ORCID}")

    # A second person is a second email address. A second display name on the
    # author's own address is the local git config, which the publication
    # recipe overrides when it rebuilds the public branch; it is reported
    # rather than failed, because that override is the thing to verify.
    ids = set()
    for line in git("log", "--all", "--format=%an <%ae>|%cn <%ce>").splitlines():
        ids.update(x.strip() for x in line.split("|") if x.strip())
    emails = {i.rsplit("<", 1)[-1].rstrip(">") for i in ids}
    foreign = sorted(e for e in emails if e != AUTHOR_EMAIL)
    if foreign:
        problems.append(f"git commits carry email addresses other than "
                        f"{AUTHOR_EMAIL}: {foreign}")
    variants = sorted(i for i in ids if i != f"{AUTHOR_NAME} <{AUTHOR_EMAIL}>"
                      and i.endswith(f"<{AUTHOR_EMAIL}>"))
    print(f"[3/5] sole author: CITATION.cff {people}, ORCID present, "
          f"{len(emails)} email address{'' if len(emails) == 1 else 'es'} "
          f"across {len(ids)} git identit{'y' if len(ids) == 1 else 'ies'}")
    for v in variants:
        print(f"  NOTE: display-name variant {v!r} on the author's own "
              f"address. PUBLISH_STEPS.md forces author and committer to "
              f"{AUTHOR_NAME!r} when it rebuilds the public branch, so this "
              f"never reaches GitHub; that forcing is load-bearing.")
    if problems:
        for p in problems:
            print("  AUTHORSHIP:", p)
        rc = stop(4, "STOPPING: the release is not solely authored by "
                     f"{AUTHOR_NAME}.")
        if rc:
            return rc
    else:
        print("      clean")

    # ---- frozen hashes ----
    sums = (ROOT / "data" / "SHA256SUMS").read_text(encoding="utf-8")
    lines = [l for l in sums.splitlines() if l.strip()]
    bad = []
    for l in lines:
        want, _, rel = l.partition(" ")
        rel = rel.strip().lstrip("*")
        f = ROOT / "data" / rel if not (ROOT / rel).exists() else ROOT / rel
        if not f.exists():
            bad.append(f"{rel}: absent")
            continue
        got = sha256(f)
        if got != want.strip():
            bad.append(f"{rel}: {got[:16]} against pinned {want.strip()[:16]}")
    print(f"[4/5] frozen hashes: {len(lines) - len(bad)} of {len(lines)} verify")
    if bad:
        for b in bad:
            print("  HASH:", b)
        rc = stop(5, "STOPPING: a frozen table does not match its pin. The "
                     "paper's numbers are read against these bytes.")
        if rc:
            return rc

    # ---- assertions ----
    if a.skip_assertions:
        print("[5/5] assertions: SKIPPED by flag. This manifest is incomplete.")
        rc_assert = None
    else:
        r = subprocess.run([sys.executable,
                            str(ROOT / "reproduction" / "reproduce_assertions.py")],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        rc_assert = r.returncode
        tail = [l for l in r.stdout.splitlines() if l.strip()][-3:]
        print(f"[5/5] assertions: exit {rc_assert}")
        for l in tail:
            print("     ", l)
        if rc_assert != 0:
            rc = stop(6, "STOPPING: the assertion harness does not exit 0.")
            if rc:
                return rc

    # ---- manifest ----
    if failed:
        print()
        print(f"DIAGNOSIS ONLY. {len(failed)} check(s) failed: "
              f"{sorted(set(failed))}. No manifest was written, because a "
              f"manifest of a tree that is not ready would read as a "
              f"clearance.")
        return max(failed)
    files, total = [], 0
    for rel in shipped:
        p = ROOT / rel
        n = p.stat().st_size
        total += n
        files.append({"path": rel, "size_bytes": n, "sha256": sha256(p)})
    man = {
        "task": "release preparation, not a release",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "author": {"name": AUTHOR_NAME, "email": AUTHOR_EMAIL, "orcid": ORCID},
        "github_main": {
            "definition": "every tracked file except those under " + EXCLUDED_PREFIX,
            "n_files": len(files), "total_bytes": total,
            "n_withheld": len(withheld),
            "untracked_working_documents": working},
        "zenodo_archive": arc,
        "preflight": {
            "every_read_artifact_tracked": True, "archive_verifies": True,
            "no_tool_attribution": True,
            "sole_author": True, "frozen_hashes_verify": len(lines),
            "assertions_exit": rc_assert},
        "head": git("rev-parse", "HEAD").strip(),
        "files": files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(man, indent=2))

    print()
    print(f"DRY RUN. Nothing was uploaded and nothing can be from this file.")
    print(f"  would ship  {len(files)} files, {total / 1e6:.1f} MB")
    print(f"  would hold back {len(withheld)} files under {EXCLUDED_PREFIX}")
    print(f"  HEAD {man['head'][:12]}")
    print(f"  -> {OUT}")
    print()
    print("The release is performed by hand, once, following PUBLISH_STEPS.md. "
          "Check the tree you push against this manifest afterwards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
