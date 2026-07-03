"""Convert Julia-style ONTS examples to JSON.

Usage:
    python instances/convert_examples.py
    python instances/convert_examples.py instances/examples --out instances/examples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from environments.onts_env import parse_onts_jl


def convert_dir(src_dir: Path, out_dir: Path) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    for src in sorted(src_dir.glob("*.jl")):
        try:
            data = parse_onts_jl(src)
        except ValueError as exc:
            skipped += 1
            print(f"skip {src.name}: {exc}")
            continue

        dst = out_dir / f"{src.stem}.json"
        dst.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        converted += 1
        print(f"wrote {dst}")
    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "src_dir",
        nargs="?",
        type=Path,
        default=Path("instances/examples"),
        help="Directory containing .jl ONTS instance files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for .json files; defaults to src_dir",
    )
    args = parser.parse_args()
    out_dir = args.out if args.out is not None else args.src_dir
    converted, skipped = convert_dir(args.src_dir, out_dir)
    print(f"converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
