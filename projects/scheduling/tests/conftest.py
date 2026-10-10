"""Support the legacy scheduling module imports from the repository root."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# HiGHS fixes its thread pool on first use in a process; later calls asking
# for a different count fail. Match the one-thread benchmark default.
os.environ.setdefault("ONTS_SOLVER_THREADS", "1")
