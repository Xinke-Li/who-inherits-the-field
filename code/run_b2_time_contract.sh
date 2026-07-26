#!/usr/bin/env bash
# Channel B2 (T2.11), detached. Build the concept event tables from the local
# OpenAlex cache, then chain the time-contract verifier. Nothing remote.
#
#   bash code/run_b2_time_contract.sh
#
# Log: results/revision/T2_11_time_contract/b2.log
set -u
cd "$(dirname "$0")/.."
mkdir -p results/revision/T2_11_time_contract
LOG=results/revision/T2_11_time_contract/b2.log
{
  echo "=== B2 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "--- step 1: concept event tables ---"
  python code/r36_concept_events.py
  rc1=$?
  echo "r36 exit $rc1"
  if [ $rc1 -ne 0 ]; then
    echo "=== B2 aborted: the event tables did not build ==="
    exit $rc1
  fi
  echo "--- step 2: time contract verification ---"
  python reproduction/verify_time_contract.py
  rc2=$?
  echo "verify_time_contract exit $rc2"
  echo "=== B2 end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$rc2 ==="
  exit $rc2
} >> "$LOG" 2>&1
