#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Incomplete studies resume their frozen source;
# completed studies regenerate reports without changing benchmark records.
export MPLCONFIGDIR="${TMPDIR:-/tmp}/onts-study-matplotlib"
study_output="${1:-projects/scheduling/research/results/local-study}"
if [[ -f "$study_output/completion.json" ]]; then
  .venv/bin/python - "$study_output" <<'PY'
import sys
from pathlib import Path
from projects.scheduling.research.report import generate_report
generate_report(Path(sys.argv[1]))
PY
  exit 0
fi
.venv/bin/python -u -m projects.scheduling.research.study \
  --output "$study_output"
