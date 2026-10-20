"""Launch recovery experiments with the hash-verified NumPy profile backend.

Direct invocation avoids importing the concurrently edited workspace backend.
The selected package and sources are copied into a temporary namespace first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--profile-snapshot", type=Path, required=True)
    parser.add_argument("--study-module", choices=["rl_recovery_study", "rl_recovery_followup", "rl_architecture_study", "rl_completion_study"], default="rl_recovery_study")
    args, remaining = parser.parse_known_args()
    snapshot = args.profile_snapshot.resolve()
    protocol = json.loads((snapshot / "protocol.json").read_text())
    backend = snapshot / "source" / "job_profile.py"
    if hashlib.sha256(backend.read_bytes()).hexdigest() != protocol["source_hashes"]["job_profile.py"]:
        raise ValueError("Profile backend snapshot mismatch")
    root = Path(__file__).resolve().parents[1]
    with TemporaryDirectory(prefix="onts-recovery-runtime-") as directory:
        base = Path(directory)
        package = base / "projects" / "scheduling"
        package.mkdir(parents=True)
        (base / "projects" / "__init__.py").write_text("")
        for source in [*root.glob("*.py"), *root.joinpath("research").glob("*.py"), root / "environments" / "onts_env.py"]:
            target = package / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(backend, package / "job_profile.py")
        sys.path.insert(0, directory)
        sys.argv = [sys.argv[0], *remaining, "--history-root", str(root / "research" / "results"),
                    "--backend-snapshot", str(snapshot)]
        import importlib
        study = importlib.import_module(f"projects.scheduling.research.{args.study_module}")
        study.main()


if __name__ == "__main__":
    main()
