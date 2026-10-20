"""Run a saved construction auditor against its verified source snapshot.

Invoke this file directly so the current workspace package is not imported.
This permits replay while another task changes the live profile backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib
import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--module", choices=["audit_rl_construct", "audit_rl_recovery"], default="audit_rl_construct")
    parser.add_argument("--auditor-source", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / "source"
    protocol = json.loads((output / "protocol.json").read_text())
    for filename, digest in protocol["source_hashes"].items():
        if hashlib.sha256((source / filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source hash mismatch: {filename}")
    with TemporaryDirectory(prefix="onts-construction-audit-") as directory:
        base = Path(directory)
        (base / "projects").mkdir()
        (base / "projects" / "__init__.py").write_text("")
        (base / "projects" / "scheduling").symlink_to(source, target_is_directory=True)
        sys.path.insert(0, directory)
        if args.auditor_source is None:
            auditor = importlib.import_module(f"projects.scheduling.research.{args.module}")
        else:
            importlib.import_module("projects.scheduling.research")
            spec = importlib.util.spec_from_file_location(f"projects.scheduling.research.{args.module}_revision", args.auditor_source.resolve())
            auditor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(auditor)
        auditor.audit(output)


if __name__ == "__main__":
    main()
