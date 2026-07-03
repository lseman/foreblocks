"""Train ONTS (Offline Nanosatellite Task Scheduling) with NCO framework.

Uses the unified NCOModel with pluggable encoders/decoders:
    - Transformer encoder + Pointer decoder (default)
    - BipartiteGNN encoder + Bipartite decoder (graph-structured)

Features:
    - MILP-shaped ONTS environment with startup/run/period/energy constraints
    - Optional bipartite graph observations with valid-inequality nodes
    - Self-evaluation (branch-and-score decoding)
    - LEHD re-encoding for large-instance generalization
    - PPO training with GAE advantage estimation

Usage:
    python examples/train_onts.py [--device cuda] [--iters 200] [--batch 64]
        [--encoder transformer|bipartite] [--self-eval-k 4] [--reencode-every 5]
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from environments.onts_env import ONTSEnv, ONTSInstancePoolEnv
from models.nco_model import build_model, register_builtin_models
from train import PPO


def main():
    parser = argparse.ArgumentParser(description="Train ONTS with NCO framework")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cpu/cuda",
    )
    parser.add_argument("--iters", type=int, default=200, help="Training iterations")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--tasks", type=int, default=9, help="Number of tasks")
    parser.add_argument("--horizon", type=int, default=100, help="Scheduling horizon")
    parser.add_argument(
        "--instance",
        type=Path,
        default=None,
        help="ONTS instance file, e.g. instances/examples/125_9.json or .jl",
    )
    parser.add_argument(
        "--instance-dir",
        type=Path,
        default=None,
        help="Directory of ONTS JSON instances to sample per episode",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="PPO epochs per update; lower is faster but noisier",
    )
    parser.add_argument(
        "--collect-per-update",
        type=int,
        default=8,
        help="Episodes collected per PPO update; lower is faster but noisier",
    )
    parser.add_argument(
        "--encoder",
        default="transformer",
        choices=["transformer", "bipartite"],
        help="Encoder type",
    )
    parser.add_argument("--d-model", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--n-layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--n-heads", type=int, default=8, help="Attention heads")
    parser.add_argument(
        "--self-eval-k",
        type=int,
        default=1,
        help="Self-eval candidates (1 = standard PPO, >1 = branch-and-score)",
    )
    parser.add_argument(
        "--self-eval-temp",
        type=float,
        default=1.5,
        help="Self-eval sampling temperature",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Curriculum: ramp k from 1 to K over training",
    )
    parser.add_argument(
        "--reencode-every",
        type=int,
        default=0,
        help="LEHD re-encoding frequency (0 = disabled)",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    torch.manual_seed(0)

    dev = args.device if torch.cuda.is_available() else "cpu"

    # Register built-in models
    register_builtin_models()

    # Create environment
    if args.instance_dir is not None:
        env = ONTSInstancePoolEnv(
            args.instance_dir,
            B=args.batch,
            device=dev,
            graph_observation=args.encoder == "bipartite",
        )
    elif args.instance is not None:
        env = ONTSEnv.from_file(
            args.instance,
            B=args.batch,
            device=dev,
            graph_observation=args.encoder == "bipartite",
        )
    else:
        env = ONTSEnv(
            B=args.batch,
            J=args.tasks,
            T=args.horizon,
            device=dev,
            graph_observation=args.encoder == "bipartite",
        )
    obs = env._obs()

    # Build model using factory
    if args.encoder == "bipartite":
        encoder_kwargs = {"edge_dim": obs["edge_features"].shape[-1]}
    else:
        encoder_kwargs = {
            "f_static": obs["task_static"].shape[-1],
            "f_dynamic": obs["task_dynamic"].shape[-1],
            "f_global": obs["glob"].shape[-1],
        }

    model = build_model(
        encoder=args.encoder,
        decoder="pointer" if args.encoder == "transformer" else "bipartite",
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        encoder_kwargs=encoder_kwargs,
        reencode_every=args.reencode_every,
    ).to(dev)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Environment: {type(env).__name__}  tasks={env.J}  horizon={env.T}")
    print(f"Model: {args.encoder} encoder + pointer decoder")
    print(f"Params: {n_params:,}   device: {dev}")
    print(
        f"Features: nodes={obs['nodes'].shape[-1]}, "
        f"static={obs['task_static'].shape[-1]}, "
        f"dynamic={obs['task_dynamic'].shape[-1]}, global={obs['glob'].shape[-1]}"
    )
    if args.encoder == "bipartite":
        print(
            f"Graph: nodes={obs['nodes'].shape[1]}, n_vars={obs['n_vars']}, "
            f"edges={obs['edge_index'].shape[1]}, edge_dim={obs['edge_features'].shape[-1]}"
        )
    if args.reencode_every > 0:
        print(f"LEHD re-encoding: every {args.reencode_every} steps")

    # Smoke test: forward + replay path
    out = model.act(obs)
    assert "new_starts" in out
    assert "logp" in out
    assert "value" in out
    # new_starts is [B, J] binary mask
    print(f"Forward + replay OK  (avg starts: {out['new_starts'].sum(1).mean():.2f})")

    # Self-eval smoke test
    if args.self_eval_k > 1:
        out_se = model.act(obs, k=args.self_eval_k, temperature=args.self_eval_temp)
        print(
            f"Self-eval OK  (k={args.self_eval_k}, avg starts: {out_se['new_starts'].sum(1).mean():.2f})"
        )

    # Train
    print(f"\nTraining for {args.iters} iterations...")
    ppo = PPO(
        model,
        device=dev,
        self_eval_k=args.self_eval_k,
        self_eval_temp=args.self_eval_temp,
        curriculum=args.curriculum,
        epochs=args.epochs,
        collect_per_update=args.collect_per_update,
    )
    hist = ppo.update(env, iters=args.iters, log_every=max(1, args.iters // 10))
    if hist:
        print(f"\nFinal return: {hist[-1]:.1f}")


if __name__ == "__main__":
    main()
