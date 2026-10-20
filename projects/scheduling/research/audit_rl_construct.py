"""Replay saved cold-start schedules, rewards, and checkpoint hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..environments.onts_env import load_onts_instance, onts_instance_to_json_data
from ..milp import ONTSConfig
from ..rl_construct import ConstructPolicy
from ..rl_construct_graph import GraphConstructPolicy
from .datasets import instance_hash
from .oracle import oracle_check


def audit(output: Path):
    def read(name):
        return json.loads((output / name).read_text())
    protocol = read("protocol.json")
    config = ONTSConfig()
    instances = {}
    for entry in protocol["manifest"]:
        instance = load_onts_instance(output / f"{entry['id']}.json")
        if instance_hash(instance) != entry["hash"]:
            raise ValueError("Instance hash mismatch")
        instances[entry["id"]] = instance
    checkpoints = read("checkpoints.json")
    for filename, digest in checkpoints.items():
        path = output / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Checkpoint hash mismatch")
        model = GraphConstructPolicy() if filename.startswith("graph-") else ConstructPolicy()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite checkpoint")
    references = {r["id"]: r for r in read("references.json")}
    evaluations, training, teachers = read("evaluations.json"), read("training.json"), read("teachers.json")
    all_rows = list(evaluations)
    for entry in [*training, *teachers]:
        all_rows.extend(dict(id=entry["id"], **r) for r in entry["rollouts"])
    for row in all_rows:
        instance = instances[row["id"]]
        valid, objective, _ = oracle_check(onts_instance_to_json_data(instance), row["schedule"], config)
        if valid != row["feasible"] or (valid and abs(objective-row["objective"]) > 1e-4):
            raise ValueError("Independent schedule replay mismatch")
        reconstructed = torch.zeros(instance.J, instance.T, dtype=torch.int)
        placed = set()
        for action in row["actions"]:
            task = action["task"]
            if task in placed:
                raise ValueError("Repeated task placement")
            placed.add(task)
            reconstructed[task] = torch.tensor(action["profile"])
        if reconstructed.tolist() != row["schedule"] or len(placed) != row["completed"]:
            raise ValueError("Action reconstruction mismatch")
        success = valid and row["runtime"] <= protocol["arguments"]["budget"]
        if success != row["success"]:
            raise ValueError("Deadline success mismatch")
        reward = 1 + objective / max(1., float(instance.priority.sum()) * instance.T) if success else .1 * len(placed) / instance.J
        if abs(reward-row["reward"]) > 1e-6:
            raise ValueError("Reward mismatch")
        if "certified_gap" in row:
            reference = references[row["id"]]
            expected = max(0., reference["objective"]-objective) / max(1., abs(reference["objective"])) if success and reference["status"] == "optimal" else None
            if (expected is None) != (row["certified_gap"] is None) or (expected is not None and abs(expected-row["certified_gap"]) > 1e-6):
                raise ValueError("Certified gap mismatch")
    for reference in references.values():
        if reference["schedule"] is not None:
            valid, objective, _ = oracle_check(onts_instance_to_json_data(instances[reference["id"]]), reference["schedule"], config)
            if not valid or abs(objective-reference["objective"]) > 1e-4:
                raise ValueError("Reference schedule mismatch")
    result = dict(validated_constructions=len(all_rows), evaluation_outputs=len(evaluations),
                  training_outputs=sum(len(t["rollouts"]) for t in training),
                  teacher_outputs=sum(len(t["rollouts"]) for t in teachers), checkpoints=len(checkpoints),
                  reference_schedules=sum(r["schedule"] is not None for r in references.values()),
                  limitations="Replay checks recorded elapsed-time credit; it cannot independently reconstruct wall-clock execution or certify solver status.")
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    audit(parser.parse_args().output)
