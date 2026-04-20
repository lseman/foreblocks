#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
VERSION_FILE = ROOT / "web" / "version.js"

text = PYPROJECT.read_text(encoding="utf-8")
match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
if not match:
    raise SystemExit("Could not find version in pyproject.toml")
version = match.group(1)
VERSION_FILE.write_text(f'window.__FOREBLOCKS_VERSION__ = "{version}";\n', encoding="utf-8")
print(f"Updated {VERSION_FILE.relative_to(ROOT)} to {version}")
