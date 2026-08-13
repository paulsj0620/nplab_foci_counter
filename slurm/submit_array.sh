#!/bin/bash
# Submit the foci_counter job array with the correct --array range for a glob.
#
# Usage (from project root):
#     bash slurm/submit_array.sh                      # all *_HE_* slides
#     bash slurm/submit_array.sh "dataset/*_HE_*.czi"  # explicit glob
#     bash slurm/submit_array.sh "dataset/*.czi" 4     # cap 4 tasks running at once
#
# The optional 2nd arg limits concurrent tasks (e.g. if GPU allocation is
# capped). Omit to let the scheduler run as many in parallel as it can.

set -euo pipefail
GLOB="${1:-dataset/*_HE_*.czi}"
CONCURRENCY="${2:-}"

FILES=( $(ls $GLOB) )          # slide names have no spaces; portable to bash 3.2
N=${#FILES[@]}
if [ "$N" -eq 0 ]; then echo "no slides matched: $GLOB"; exit 1; fi

RANGE="0-$((N - 1))"
[ -n "$CONCURRENCY" ] && RANGE="${RANGE}%${CONCURRENCY}"

echo "submitting $N slides as array $RANGE for glob: $GLOB"
sbatch --array="$RANGE" slurm/run_one.slurm "$GLOB"

echo
echo "monitor:   squeue -u \$USER"
echo "after all tasks finish, build the combined workbook (fast, cache-aware):"
echo "   python scripts/run_batch.py \"$GLOB\""
