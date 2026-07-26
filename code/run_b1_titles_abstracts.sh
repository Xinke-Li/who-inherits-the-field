#!/usr/bin/env bash
# Channel B1, detached. One read-only OpenAlex fetch serving T3.1 and T3.2,
# then the per-field merge. The key is read from .openalex_key inside the
# python script and is never printed or written; this launcher never touches it.
#
#   bash code/run_b1_titles_abstracts.sh
#
# Resumable: re-running skips authors already in data/supplement/_ta_cache/.
# Log: results/revision/T3_1_supplement/b1.log
set -u
cd "$(dirname "$0")/.."
mkdir -p results/revision/T3_1_supplement
LOG=results/revision/T3_1_supplement/b1.log

# Detach stdin here rather than at the call site. A launcher that inherits a
# terminal dies with it, and wrapping the call in "nohup ... < /dev/null" from
# PowerShell failed to start at all, so the detachment belongs in the script
# where it can be tested.
exec < /dev/null

# A detached launcher does not inherit the interactive shell's PATH, and a bare
# "python" there resolves to the Windows Store stub, which exits 49 without
# running anything. Resolve the interpreter explicitly and refuse if it is the
# stub rather than fetching nothing for four hours.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in "$HOME/anaconda3/python.exe" "$HOME/miniconda3/python.exe" \
              "$(command -v python3 || true)" "$(command -v python || true)"; do
    if [ -n "$cand" ] && [ -x "$cand" ] && "$cand" -c "import pandas" 2>/dev/null; then
      PY="$cand"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "no usable python found; set PYTHON=/path/to/python" >> "$LOG"
  exit 2
fi

# A second concurrent fetcher is worse than none. Two ran at once earlier: each
# carries its own 8 requests/second limiter, so the pair pushed about 16 at the
# polite pool, and each derives its next shard number from the shard COUNT at
# its own start, so as the counters converge one silently overwrites the
# other's shard and that work is lost. Refuse rather than double up.
LOCK=data/supplement/_ta_cache/.fetch.lock
mkdir -p data/supplement/_ta_cache
if [ -f "$LOCK" ]; then
  OLD=$(cat "$LOCK" 2>/dev/null || echo "")
  if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
    echo "=== B1 refused $(date -u +%Y-%m-%dT%H:%M:%SZ): already running as PID $OLD ===" >> "$LOG"
    exit 3
  fi
  echo "stale lock from PID $OLD, taking over" >> "$LOG"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

{
  echo "=== B1 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "interpreter: $PY  launcher PID $$"
  echo "--- step 1: fetch (resumable) ---"
  "$PY" code/r35_fetch_titles_abstracts.py
  rc1=$?
  echo "fetch exit $rc1"
  if [ $rc1 -ne 0 ]; then
    echo "=== B1 aborted in the fetch; re-run this script to resume ==="
    exit $rc1
  fi
  echo "--- step 2: merge into per-field files ---"
  "$PY" code/r35_fetch_titles_abstracts.py --merge
  rc2=$?
  echo "merge exit $rc2"
  echo "=== B1 end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc2 ==="
  exit $rc2
} >> "$LOG" 2>&1
