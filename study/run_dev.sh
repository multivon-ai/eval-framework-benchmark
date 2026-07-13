#!/usr/bin/env bash
# Run ALL 30 dev cells sequentially (plan §6 dev row: 5 frameworks x 2
# judges x 3 tasks, run 0). Per-cell logs in study/runs/logs/. A failing
# cell is logged and skipped (continue-on-cell-failure); the summary at the
# end lists failures and the script exits non-zero if any occurred.
#
# Env first:  set -a; . ~/Documents/.env.local; set +a
# Usage:      bash study/run_dev.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$REPO/.venv-study/bin/python}"
LOG_DIR="$REPO/study/runs/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: $PYTHON not found — build .venv-study per study/README.md" >&2
    exit 1
fi

TASKS=(ragtruth-sum halueval-sum halueval-qa)
JUDGES=(gpt-4o-mini claude-haiku-4-5)
FRAMEWORKS=(multivon-eval deepeval ragas trulens opik)

failed=()
n=0
total=$(( ${#TASKS[@]} * ${#JUDGES[@]} * ${#FRAMEWORKS[@]} ))

for task in "${TASKS[@]}"; do
  for judge in "${JUDGES[@]}"; do
    for fw in "${FRAMEWORKS[@]}"; do
      n=$((n + 1))
      cell="${task}_dev_${fw}_${judge}_run0"
      log="$LOG_DIR/${cell}.log"
      echo "=== [$n/$total] $cell (log: $log)"
      # The if-guard keeps set -e from killing the loop on a cell failure:
      # log it, record it, move on to the next cell.
      if ! "$PYTHON" "$REPO/study/run_study.py" \
            --task "$task" --split dev --framework "$fw" \
            --judge "$judge" --run 0 >"$log" 2>&1; then
        echo "    CELL FAILED: $cell — see $log" >&2
        failed+=("$cell")
      else
        tail -n 1 "$log"
      fi
    done
  done
done

echo
echo "=== dev run summary: $((total - ${#failed[@]}))/$total cells succeeded"
if (( ${#failed[@]} > 0 )); then
  echo "FAILED cells:" >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
echo "all dev cells complete — check status with:"
echo "  $PYTHON $REPO/study/run_study.py --cell-list"
