"""Independent schedule/reward replay and rejection-trace replay for recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import load_onts_instance, onts_instance_to_json_data
from ..milp import ONTSConfig
from ..rl_construct import ConstructionEnv, Placement
from ..rl_recovery import RecoveryPolicy, completion_probe
from .datasets import instance_hash
from .oracle import oracle_check


def audit(output, rejection_limit=512):
    output = Path(output)
    def read(name):
        return json.loads((output/name).read_text())
    protocol = read("protocol.json"); config = ONTSConfig(**protocol["config"])
    inputs = {}
    for entry in protocol["manifest"]:
        instance = load_onts_instance(output/f"{entry['id']}.json")
        if instance_hash(instance) != entry["hash"]:
            raise ValueError("Instance hash mismatch")
        inputs[entry["id"]] = instance
    checkpoints = read("checkpoints.json")
    for filename, digest in checkpoints.items():
        path = output/filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Checkpoint hash mismatch")
        if protocol.get("completion_critic"):
            from ..rl_architectures import ArchitecturePolicy
            from ..rl_completion import CompletionPolicy
            model = CompletionPolicy(protocol["arguments"]["width"],protocol["arguments"]["guidance"]) if filename.startswith("critic-") else ArchitecturePolicy("mlp",protocol["arguments"]["width"])
        elif "architectures" in protocol:
            from ..rl_architectures import ArchitecturePolicy
            model = ArchitecturePolicy(filename.split("-")[0], protocol["arguments"]["width"])
        else:
            model = RecoveryPolicy()
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite checkpoint")
    rows = [(r, protocol["arguments"]["eval_budget"]) for r in read("evaluations.json")]
    for entry in read("training.json"):
        rows.extend((dict(id=entry["id"], **r), protocol["arguments"]["train_budget"]) for r in entry["rollouts"])
    for entry in read("teachers.json"):
        rows.extend((dict(id=entry["id"], **r), protocol["arguments"]["teacher_budget"]) for r in entry["rollouts"])
    completed_schedules = rejection_checks = placed_checks = skipped_rejections = 0
    rejection_cache = {}
    for row, budget in rows:
        instance = inputs[row["id"]]; data = onts_instance_to_json_data(instance)
        valid, objective, _ = oracle_check(data, row["schedule"], config)
        if valid != row["feasible"] or (valid and abs(objective-row["objective"]) > 1e-4):
            raise ValueError("Independent final schedule disagreement")
        expected = valid and row["runtime"] <= budget
        if expected != row["success"]:
            raise ValueError("Deadline success mismatch")
        depth = row.get("deepest_completed", row["completed"])
        reward = 1+objective/max(1., float(instance.priority.sum())*instance.T) if expected else .1*depth/instance.J
        if abs(reward-row["reward"]) > 1e-6:
            raise ValueError("Reward mismatch")
        final = np.zeros((instance.J, instance.T), dtype=int); final_tasks = set()
        for action in row["actions"]:
            if action["task"] in final_tasks:
                raise ValueError("Repeated final task")
            final_tasks.add(action["task"]); final[action["task"]] = action["profile"]
        if final.tolist() != row["schedule"] or len(final_tasks) != row["completed"]:
            raise ValueError("Final action reconstruction mismatch")
        if "events" not in row:
            continue
        path, credited = [], []
        seen = {(np.zeros((instance.J, instance.T), dtype=bool).tobytes(), tuple(range(instance.J)))}
        last_elapsed = 0.
        for event in row["events"]:
            if event["elapsed"] < last_elapsed or event["elapsed"] > row["runtime"]+1e-6:
                raise ValueError("Nonmonotonic event timestamps")
            last_elapsed = event["elapsed"]
            if event["kind"] == "proposal":
                path = path[:event["depth"]]
                if [a["task"] for a in path] != event["prefix"]:
                    raise ValueError("Prefix reconstruction mismatch")
                trial = np.zeros((instance.J, instance.T), dtype=bool)
                for action in path:
                    trial[action["task"]] = action["profile"]
                probe = event["probe"]
                candidate_state = trial.copy(); candidate_state[event["task"]] = event["profile"]
                key_state = (candidate_state.tobytes(), tuple(sorted(set(range(instance.J))-{a["task"] for a in path}-{event["task"]})))
                if probe["status"] == "duplicate_state" and key_state not in seen:
                    raise ValueError("Memoized state was never explored")
                if probe["status"] == "reject":
                    key = (row["id"], trial.tobytes(), event["task"], tuple(event["profile"]))
                    if key not in rejection_cache and (rejection_limit is None or len(rejection_cache) < rejection_limit):
                        env = ConstructionEnv(instance, config); env.schedule = trial
                        env.remaining = set(range(instance.J))-{a["task"] for a in path}
                        replay = completion_probe(env, Placement(event["task"], np.array(event["profile"], dtype=bool), []))
                        if replay["status"] != "reject" or replay["reason"] != probe["reason"]:
                            raise ValueError("Necessary-condition rejection mismatch")
                        rejection_cache[key] = True
                    if key in rejection_cache:
                        rejection_checks += 1
                    else:
                        skipped_rejections += 1
                if event.get("placed"):
                    if row.get("memoize") and key_state in seen:
                        raise ValueError("Duplicate state was explored despite memoization")
                    seen.add(key_state)
                    task = event["task"]
                    if task in {a["task"] for a in path}:
                        raise ValueError("Repeated partial task")
                    task_data = {name: [values[task]] if name in ("priority", "uso_p", "min_statup", "max_statup", "min_cpu_time", "max_cpu_time", "min_periodo_job", "max_periodo_job", "win_min", "win_max") else values for name, values in data.items()}
                    if not oracle_check(task_data, [event["profile"]], config)[0]:
                        raise ValueError("Placed profile violates task constraints")
                    trial[task] = event["profile"]
                    usage = instance.uso_p.cpu().double().numpy() @ trial
                    solar = instance.recurso_p.cpu().double().numpy(); charge = config.soc_inicial
                    for load, incoming in zip(usage, solar):
                        charge = min(1., charge+(incoming-load)*config.ef/(60*config.q*config.v_bat))
                        if load > incoming+config.bat_usage*config.v_bat+1e-5 or charge < config.limite_inferior-1e-5:
                            raise ValueError("Placed prefix violates resources")
                    path.append(dict(task=task, profile=event["profile"])); placed_checks += 1
            elif event["kind"] == "complete":
                schedule = np.zeros((instance.J, instance.T), dtype=int)
                for action in event["actions"]:
                    schedule[action["task"]] = action["profile"]
                feasible, value, _ = oracle_check(data, schedule.tolist(), config)
                if feasible != event["feasible"] or abs(value-event["objective"]) > 1e-4:
                    raise ValueError("Independent completion disagreement")
                if event["credited"]:
                    if not feasible or event["elapsed"] > row["cutoff"]+1e-6:
                        raise ValueError("Late or infeasible completion credit")
                    credited.append((value, schedule.tolist()))
                completed_schedules += 1
            elif event["kind"] != "backtrack":
                raise ValueError("Unknown event")
        if valid:
            initial_valid, initial_objective, _ = oracle_check(data, np.zeros_like(final).tolist(), config)
            eligible = [v for v, _ in credited] + ([initial_objective] if initial_valid else [])
            if not eligible or abs(max(eligible)-objective) > 1e-4:
                raise ValueError("Best feasible output was not preserved")
            if not any(schedule == row["schedule"] for value, schedule in credited if abs(value-objective) < 1e-4) and not (initial_valid and not np.any(final)):
                raise ValueError("Returned schedule has no timely validation event")
            if row["time_to_first_feasible"] is None or row["time_to_first_feasible"] > row["cutoff"]+1e-6:
                raise ValueError("Invalid time to first feasible output")
        if row["rejected"] != sum(e["kind"] == "proposal" and e["probe"]["status"] == "reject" for e in row["events"]):
            raise ValueError("Rejection count mismatch")
        if row["backtracks"] != sum(e["kind"] == "backtrack" for e in row["events"]):
            raise ValueError("Backtrack count mismatch")
    for reference in read("references.json"):
        if reference["schedule"] is not None:
            valid, objective, _ = oracle_check(onts_instance_to_json_data(inputs[reference["id"]]), reference["schedule"], config)
            if not valid or abs(objective-reference["objective"]) > 1e-4:
                raise ValueError("Reference replay mismatch")
    result = dict(constructions=len(rows), completed_schedules=completed_schedules, placed_prefixes=placed_checks,
                  replayed_rejections=rejection_checks, unique_rejections=len(rejection_cache), checkpoints=len(checkpoints),
                  skipped_rejection_replays=skipped_rejections, rejection_limit=rejection_limit,
                  limitations="All recorded schedules, placements, rewards, deadlines and memoization traces are checked. Rejection DP replay is bounded to the first distinct states up to rejection_limit; it uses the frozen production DP. Runtime and solver status are not independent certificates.")
    if "architectures" in protocol:
        from ..rl_architectures import ArchitecturePolicy, observation
        snapshot_checks = 0
        for row,_ in rows:
            env = ConstructionEnv(inputs[row["id"]],config)
            for action,decision in zip(row["actions"],row.get("decisions",[])):
                raw = decision["observation"]; index = decision["index"]
                if raw["task_indices"][index] != action["task"] or raw["profiles"][index] != action["profile"] or raw["features"] != decision["features"]:
                    raise ValueError("Teacher action/snapshot mismatch")
                choices = [Placement(int(task),np.array(profile,dtype=bool),features[:16]) for task,profile,features in zip(raw["task_indices"],raw["profiles"],raw["features"])]
                expected = observation(env,choices,torch.tensor(decision["features"]))
                for key,value in expected.items():
                    if not torch.allclose(value,torch.tensor(raw[key],dtype=value.dtype),atol=1e-6):
                        raise ValueError("Teacher snapshot is not the pre-placement state")
                env.place(choices[index]); snapshot_checks += 1
        solver_rows = read("solver_evaluations.json")
        for row in solver_rows:
            instance = inputs[row["id"]]
            feasible,value,_ = oracle_check(onts_instance_to_json_data(instance),row["schedule"],config)
            success = feasible and row["runtime"] <= protocol["arguments"]["eval_budget"]
            utility = 1+value/max(1.,float(instance.priority.sum())*instance.T) if success else 0.
            if feasible != row["feasible"] or success != row["success"] or abs(utility-row["utility"])>1e-6 or (feasible and abs(value-row["objective"])>1e-4):
                raise ValueError("MILP baseline replay mismatch")
        result.update(teacher_snapshots=snapshot_checks,solver_baselines=len(solver_rows))
    if protocol.get("completion_critic"):
        from ..rl_completion import branch_labels
        evidence_checks=0
        for row,_ in rows:
            expected=branch_labels(row)
            for e in row.get("completion_evidence",[]):
                if expected.get(e["event_index"]) != e["label"]:
                    raise ValueError("Completion evidence label mismatch")
                evidence_checks+=1
            trace=row.get("learning_trace",[])
            if trace and len(trace)!=row["sampled_decisions"]:
                raise ValueError("Learning trace count mismatch")
            for record in trace:
                if expected.get(record["event_index"])!=record["completion_target"] or not np.isfinite(record["old_value"]) or not np.isfinite(record["old_log_prob"]):
                    raise ValueError("Learning trace mismatch")
        teacher_evidence=read("teacher_evidence.json")
        teachers=read("teachers.json")
        for entry in teacher_evidence:
            row=next(t for t in teachers if t["id"]==entry["id"])["rollouts"][entry["attempt"]]
            expected=branch_labels(row)
            for e in entry["examples"]:
                if expected.get(e["event_index"])!=e["label"]:raise ValueError("Teacher completion label mismatch")
                evidence_checks+=1
        update_rows=read("updates.json")
        for update in update_rows:
            for name in ("ppo","control"):
                info=update[name]
                if info["updated"] and info["actor_anchor_kl"]>protocol["arguments"]["kl_limit"]+1e-6:
                    raise ValueError("Accepted actor update exceeds anchor KL")
        result.update(completion_evidence=evidence_checks,conservative_updates=len(update_rows),
            completion_audit_limitations="Labels are reconstructed from independently schedule-checked events using the frozen labeling procedure; exhaustive subtree semantics and numerical update replay are not independently verified. Unknown completion labels are not forced to negative.")
    write_path = output/"audit.json"; write_path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejection-limit", type=int, default=512)
    args = parser.parse_args()
    audit(args.output, None if args.rejection_limit == 0 else args.rejection_limit)
