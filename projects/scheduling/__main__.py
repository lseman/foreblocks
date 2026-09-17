"""Run scheduling as a package: python -m scheduling"""

import sys
from pathlib import Path

# Ensure scheduling/ is on the path so subpackages resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))
