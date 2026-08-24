#!/bin/bash
# rig-ops acceptance runner: run deliverables_check.py over every rigged glb.
# Writes per-role PASS/FAIL summaries to /tmp/accept_all.log
cd ~/work/code || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch2.4_cuda12.1
LOG=/tmp/accept_all.log
: > "$LOG"
for f in ~/work/outputs/*_rigged.glb; do
  name=$(basename "$f" _rigged.glb)
  out=/tmp/accept_one.log
  PYTHONPATH=. python scripts/deliverables_check.py "$f" > "$out" 2>&1
  rc=$?
  echo "### $name rc=$rc" >> "$LOG"
  cat "$out" >> "$LOG"
  echo "[accepted: $name rc=$rc]"
done
echo "ACCEPT_ALL_DONE"
