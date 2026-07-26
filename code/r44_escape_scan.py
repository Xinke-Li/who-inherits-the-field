#!/usr/bin/env python3
r"""Scan the LaTeX sources for backslash commands eaten by an escape expansion.

THE BUG CLASS. When a LaTeX command passes through a shell heredoc or an
unescaped Python string, a backslash followed by one of r n t b f v a is
expanded to the control character it names. `\ref` becomes a carriage return
plus `ef`, `\newpage` becomes a newline plus `ewpage`, `\textbf` becomes a tab
plus `extbf`. What survives in the file is an orphan fragment, and LaTeX
typesets it as literal text. There is no warning of any kind: a mangled `\ref`
is not an undefined reference, so a build reports zero undefined references
while printing `ef{sec:limits}` into the PDF.

One instance was found by hand in main.tex on 2026-07-25 and repaired. This
scans for the whole class, in every file the paper builds, and is wired into
reproduce_assertions.py so a regenerated fragment cannot reintroduce it
silently. The generated fragments are the highest risk, because they come out
of the same emitter path that produced the original.

  python code/r44_escape_scan.py             # scan paper/, exit 1 on any hit
  python code/r44_escape_scan.py --pdf       # also read the PDF text layer

A hit is reported with file, line, column and the raw bytes around it, and is
classified. Some fragments are ordinary English words: `eta`, `ho`, `space`,
`title`. The classifier separates them by asking whether the fragment sits
where a control sequence would have to be, not by asking whether the letters
appear.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

# Every LaTeX command whose first letter is one of the seven expanded escapes,
# with that first letter stripped. This is the user-supplied list plus the
# commands actually used in this project's sources.
FRAGMENTS = [
    # \r -> carriage return
    "ef{", "efs{", "enewcommand", "aggedbottom", "aggedright", "ule{",
    "ightarrow", "ho", "m{", "esizebox", "ow{", "elax",
    # \n -> newline
    "ewcommand", "ewpage", "oindent", "ewline", "ormalsize", "umberline",
    "u{", "ewif", "ewenvironment",
    # \t -> tab
    "extbf{", "extit{", "exttt{", "extsc{", "extrm{", "extwidth", "extcolor",
    "ext{", "able", "oprule", "op{", "imes", "ilde{", "ag{", "itle{", "au{",
    "abular", "abularx", "hispage", "heta",
    # \b -> backspace
    "ottomrule", "egin{", "ibliography", "ibitem", "aselineskip", "f{",
    "ig{", "reak", "ox{",
    # \f -> form feed
    "rac{", "ootnote", "ootnotesize", "ill", "igure", "lushleft", "box{",
    # \v -> vertical tab
    "space{", "arepsilon", "arphi", "artheta", "ec{", "fill", "erb",
    # \a -> bell
    "lpha", "bstract", "ppendix", "rraystretch", "uthor", "lign", "nd{",
    "ddtolength", "boverule",
]
# Sorted longest first so the most specific fragment is reported.
FRAGMENTS = sorted(set(FRAGMENTS), key=len, reverse=True)

CONTROL = {"\t": "TAB U+0009", "\r": "CR U+000D", "\f": "FF U+000C",
           "\v": "VT U+000B", "\b": "BS U+0008", "\a": "BEL U+0007"}

# A real orphan is never followed by another lowercase letter, because the rest
# of the command's name was consumed with the backslash: `\rho` leaves `ho` and
# then whatever followed the command, never `honest`. That one condition is
# what separates the fragment `ho` from the words honest, holds, how and hoc,
# and `fill` from filled. Fragments ending in a brace are unambiguous already.
# The lookahead applies only to fragments that do NOT end in a brace. A brace
# is always followed by its argument, so `ef{sec:limits}` must not be rejected
# for having a letter after it. Getting this wrong made the scanner miss three
# of the seven escape classes; the positive control in
# scratchpad/escape_control.py is what caught that.
BRACE = [f for f in FRAGMENTS if f.endswith("{")]
WORDY = [f for f in FRAGMENTS if not f.endswith("{")]
ALT = ("(?:(?:" + "|".join(re.escape(f) for f in BRACE) + ")"
       "|(?:(?:" + "|".join(re.escape(f) for f in WORDY) + ")(?![a-z])))")

# An orphan at line start is the signature of \r or \n expansion, because the
# control character became the line break itself. An orphan after a control
# character on the same line is the signature of \t \b \f \v \a.
ORPHAN_AT_START = re.compile(r"^[ \t]*(" + ALT + r")")

# TeX consumes braces, so a mangled \ref renders as `Sectionefsec:limits`, not
# as `ef{sec:limits}`. Searching the text layer for the source fragment can
# therefore never match. What a correct PDF never contains is a label key, so
# that is what the PDF is searched for, alongside the brace-free fragments.
LABEL_IN_PDF = re.compile(
    r"(?:sec|tab|fig|app|eq|alg|thm|lem|def|fn|itm):[A-Za-z0-9_]{2,}")
ORPHAN_IN_PDF = re.compile(
    r"(?<![A-Za-z\\])((?:" + "|".join(re.escape(f) for f in WORDY) +
    r")(?![a-z]))")


def tex_files():
    """Every .tex file the paper builds: main plus everything it pulls in."""
    main = PAPER / "main.tex"
    if not main.is_dir() and main.exists():
        src = main.read_text(encoding="utf-8", errors="replace")
        pulled = re.findall(r"\\(?:input|include)\{([^}]*)\}", src)
        out = [main]
        for p in pulled:
            f = PAPER / (p if p.endswith(".tex") else p + ".tex")
            if f.exists():
                out.append(f)
        # anything else in paper/ that is not a class or style file
        for f in sorted(PAPER.glob("*.tex")):
            if f not in out:
                out.append(f)
        return out
    return []


def classify(fragment, line, col, prev_line):
    """Real corruption, or an ordinary word that happens to match?

    The test is positional, not lexical. A fragment is a real corruption when
    the character position it occupies is one a control sequence would have to
    occupy: immediately after a control character, or at the start of a line
    whose predecessor ends where a command would have been interrupted.
    """
    before = line[:col]
    if before and before[-1] in CONTROL:
        return "REAL", f"preceded by {CONTROL[before[-1]]} on the same line"
    if col == 0 or before.strip() == "":
        p = (prev_line or "").rstrip()
        # A command interrupted mid-token leaves the previous line ending in a
        # tie, an opening brace, a tilde, or simply mid-sentence with no
        # terminal punctuation and no closing brace.
        if p.endswith(("~", "{", "(", "[", "$", ",")):
            return "REAL", f"previous line ends {p[-1]!r}, mid-command"
        if p and not p.endswith((".", ":", ";", "}", "%", "\\\\")):
            return "SUSPECT", "line-initial fragment after an unterminated line"
        return "SUSPECT", "line-initial fragment"
    return "WORD", "ordinary text position"


def scan_text(path, src):
    hits = []
    lines = src.split("\n")
    for i, line in enumerate(lines):
        prev = lines[i - 1] if i else ""

        # 1. orphan fragments at line start or after whitespace
        m = ORPHAN_AT_START.match(line)
        if m:
            kind, why = classify(m.group(1), line, m.start(1), prev)
            hits.append((path, i + 1, m.start(1) + 1, m.group(1), kind, why,
                         line[:90]))

        # 2. orphan fragments straight after a control character anywhere
        for ch in CONTROL:
            for c in [j for j, x in enumerate(line) if x == ch]:
                rest = line[c + 1:]
                m2 = re.match(r"(" + ALT + r")", rest)
                if m2:
                    hits.append((path, i + 1, c + 2, m2.group(1), "REAL",
                                 f"immediately after {CONTROL[ch]}",
                                 line[:90]))

        # 3. any control character on a line that contains a backslash
        if "\\" in line:
            for ch, name in CONTROL.items():
                if ch in line and ch != "\t":
                    hits.append((path, i + 1, line.index(ch) + 1, repr(ch),
                                 "REAL", f"{name} inside a line with a "
                                 f"backslash", line[:90]))

        # 4. a line ending in a bare backslash (not the \\ row break)
        if re.search(r"(?<!\\)\\$", line):
            hits.append((path, i + 1, len(line), "\\", "REAL",
                         "line ends in a bare backslash", line[:90]))
    return hits


def scan_pdf():
    """Read the PDF's own text layer. Do not reason about it from the source."""
    pdf = PAPER / "main.pdf"
    if not pdf.exists():
        return None, "paper/main.pdf is absent"
    txt = None
    try:
        import subprocess
        r = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0 and r.stdout.strip():
            txt = r.stdout
    except (OSError, FileNotFoundError):
        pass
    if txt is None:
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                return None, ("no PDF text extractor available; install pypdf "
                              "or put pdftotext on PATH")
        txt = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages)
    found = []
    for m in LABEL_IN_PDF.finditer(txt):
        ctx = txt[max(0, m.start() - 45):m.end() + 45].replace("\n", " ")
        found.append(("label key " + m.group(0), ctx))
    for m in ORPHAN_IN_PDF.finditer(txt):
        ctx = txt[max(0, m.start() - 45):m.end() + 45].replace("\n", " ")
        found.append((m.group(1), ctx))
    return found, f"{len(txt)} characters of text layer"


FRAGILE = ("path", "url", "verb")


def fragile_in_captions():
    """Every fragile command sitting unprotected inside a moving argument.

    A caption is a moving argument: LaTeX writes it out a second time, and a
    command that redefines its own catcodes cannot survive the trip. The error
    it produces names \\url even when the source says \\path, and it appears
    only once the caption is written, which is why it can pass a draft build and
    fail the one that counts. Two captions written in one session hit it. The
    remedy the paper already uses everywhere else is \\protect, so this makes
    the convention checkable rather than remembered.

    Returns a list of (file, line, command) for unprotected occurrences.
    """
    out = []
    for f in tex_files():
        src = f.read_text(encoding="utf-8", errors="replace")
        i = 0
        while True:
            j = src.find("\\caption{", i)
            if j < 0:
                break
            k, depth = j + len("\\caption{"), 1
            while depth and k < len(src):
                if src[k] == "{":
                    depth += 1
                elif src[k] == "}":
                    depth -= 1
                k += 1
            body = src[j:k]
            for cmd in FRAGILE:
                token = "\\" + cmd + "{"
                p = 0
                while True:
                    q = body.find(token, p)
                    if q < 0:
                        break
                    if not body[:q].endswith("\\protect"):
                        line = src[:j + q].count("\n") + 1
                        out.append((f.name, line, "\\" + cmd))
                    p = q + len(token)
            i = k
    return out


def run():
    """Return (real_hits, suspect_hits, files_scanned). Importable."""
    files = tex_files()
    real, suspect = [], []
    for f in files:
        for h in scan_text(f.name, f.read_text(encoding="utf-8",
                                               errors="replace")):
            (real if h[4] == "REAL" else suspect).append(h)
    return real, suspect, [f.name for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()

    real, suspect, files = run()
    if not files:
        print("STOPPING: no .tex files found under paper/. In a public clone "
              "that is expected, because paper/ is not part of the release.")
        return 2

    print(f"[r44] scanned {len(files)} files: {', '.join(files)}")
    print(f"[r44] {len(FRAGMENTS)} orphan fragments, "
          f"{len(CONTROL)} control characters")
    print()
    for label, hits in (("REAL", real), ("SUSPECT", suspect)):
        print(f"--- {label}: {len(hits)} ---")
        for path, ln, col, frag, kind, why, ctx in hits:
            print(f"  {path}:{ln}:{col}  {frag!r}  {why}")
            print(f"      {ctx!r}")
    if not real and not suspect:
        print("  clean: no orphaned fragment, no control character beside a "
              "backslash, no line ending in a bare backslash")

    if a.pdf:
        print()
        found, note = scan_pdf()
        if found is None:
            print(f"--- PDF: not read, {note} ---")
        else:
            print(f"--- PDF text layer: {note} ---")
            if found:
                for f, ctx in found:
                    print(f"  ORPHAN IN PDF {f!r}: ...{ctx}...")
            else:
                print("  clean: no orphan fragment in the rendered text")

    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main())
