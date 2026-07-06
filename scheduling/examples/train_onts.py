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


@torch.no_grad()
def evaluate(model, env, *, plot_path: Path | None = None) -> dict:
    """Run one greedy rollout on `env` (expects B=1) and report feasibility.

    Returns a dict with total return, per-task start counts vs.
    min/max_statup bounds, and any SoC (energy) violations recorded
    during the rollout.
    """
    model.eval()
    obs = env.reset()
    total_r = torch.zeros(env.B, device=env.dev)
    soc_trace = []
    power_trace = []
    x_trace = []
    while obs is not None:
        out = model.act(obs, temperature=0.0)
        nobs, r, done = env.step(out["new_starts"])
        total_r += r
        soc_trace.append(env.soc.clone())
        power_trace.append(env.power[:, env.t - 1].clone())
        x_trace.append(env.active.clone())
        obs = nobs if not done else None

    starts = env.starts[0]
    min_statup = env.min_statup
    max_statup = env.max_statup
    unmet = (min_statup - starts).clamp(min=0)
    over = (starts - max_statup).clamp(min=0)
    soc_stack = torch.stack(soc_trace, dim=1)[0]  # [T]
    soc_violation = (soc_stack < env.limite_inferior - 1e-6).any().item()

    feasible = bool((unmet == 0).all() and (over == 0).all() and not soc_violation)

    report = {
        "return": total_r[0].item(),
        "feasible": feasible,
        "unmet_min_startups": unmet.tolist(),
        "over_max_startups": over.tolist(),
        "soc_violation": soc_violation,
    }

    print(f"\n=== Evaluation ({'instance' if plot_path else 'env'}) ===")
    print(f"Return: {report['return']:.2f}")
    print(f"Feasible: {report['feasible']}")
    if any(unmet.tolist()):
        print(f"  Unmet min startups per task: {unmet.tolist()}")
    if any(over.tolist()):
        print(f"  Over max startups per task: {over.tolist()}")
    if soc_violation:
        print("  SoC dropped below limite_inferior at some timestep")

    if plot_path is not None:
        _plot_gantt(
            env,
            torch.stack(x_trace, dim=-1)[0],  # [J, T]
            torch.stack(power_trace, dim=-1)[0],  # [T]
            soc_stack,
            plot_path,
        )
        print(f"Gantt chart saved to {plot_path}")

    model.train()
    return report


def _plot_gantt(env, x, power, soc, path: Path) -> None:
    """Plot solar/task power + SoC, and per-task on/off bars, like original.py."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    T, J = env.T, env.J
    solar = env.recurso_p.cpu()
    power = power.cpu()
    soc = soc.cpu()
    x = x.cpu()

    fig, (ax_power, *axs) = plt.subplots(
        1 + J, 1, figsize=(10, 2 + J), gridspec_kw={"height_ratios": [3] + [1] * J}
    )

    ax_power.plot(solar, label="Solar panel power")
    ax_power.plot(power, color="gray", label="Task consumption")
    ax_power.fill_between(range(T), solar, power, color="black", alpha=0.05)
    ax_power.set_ylabel("Power [W]")
    ax_power.legend(loc="upper left", fontsize=7)
    ax_soc = ax_power.twinx()
    ax_soc.plot(soc * 100, "k", label="SoC [%]")
    ax_soc.set_ylim(0, 100)
    ax_soc.legend(loc="upper right", fontsize=7)
    ax_power.set_xlim(0, T - 1)

    cmap = plt.get_cmap("viridis")
    for j in range(J):
        axs[j].fill_between(range(T), x[j].float(), step="post", color=cmap(j / J))
        axs[j].set_ylim(0, 1)
        axs[j].set_xlim(0, T - 1)
        axs[j].set_yticks([])
        axs[j].set_ylabel(f"Job {j}", rotation=0, ha="right", va="center", fontsize=7)
    axs[-1].set_xlabel("Time [min]")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--eval-instance",
        type=Path,
        default=None,
        help="Instance file (.json/.jl) to evaluate on after training. "
        "Defaults to --instance, or a file from --instance-dir, or a "
        "fresh random instance of the same size, if not given",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs"),
        help="Directory for auto-generated eval outputs (Gantt chart)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Path to save a Gantt chart (power/SoC + per-task schedule) for "
        "the eval rollout. Defaults to <outdir>/gantt.png",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic post-training evaluation and plot",
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

    if not args.no_eval:
        args.outdir.mkdir(parents=True, exist_ok=True)
        plot_path = args.plot if args.plot is not None else args.outdir / "gantt.png"

        if args.eval_instance is not None:
            eval_env = ONTSEnv.from_file(
                args.eval_instance,
                B=1,
                device=dev,
                graph_observation=args.encoder == "bipartite",
            )
        elif args.instance is not None:
            eval_env = ONTSEnv.from_file(
                args.instance,
                B=1,
                device=dev,
                graph_observation=args.encoder == "bipartite",
            )
        elif args.instance_dir is not None:
            eval_path = env.instance_paths[0]
            eval_env = ONTSEnv.from_file(
                eval_path,
                B=1,
                device=dev,
                graph_observation=args.encoder == "bipartite",
            )
        else:
            eval_env = ONTSEnv(
                B=1,
                J=args.tasks,
                T=args.horizon,
                device=dev,
                graph_observation=args.encoder == "bipartite",
                seed=1,
            )

        evaluate(model, eval_env, plot_path=plot_path)


if __name__ == "__main__":
    main()
