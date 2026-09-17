"""Diagnose why training doesn't learn."""

import torch
import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, "/data/dev/foreblocks/scheduling")
from environments.onts_env import ONTSEnv
from models.nco_model import build_model, register_builtin_models
from train import PPO

torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {dev}")
register_builtin_models()

env = ONTSEnv(B=8, J=9, T=100, device=dev)
model = build_model(
    encoder="transformer",
    decoder="pointer",
    d_model=128,
    n_heads=8,
    n_layers=3,
    encoder_kwargs={"f_static": 10, "f_dynamic": 10, "f_global": 4},
).to(dev)
model.eval()

# 1. Check what model picks
print("=== 1. Action Distribution ===")
for i in range(5):
    obs = env.reset()
    with torch.no_grad():
        out = model.act(obs, temperature=10.0)
    picks = out["new_starts"]
    n_tasks = picks.sum(1).tolist()
    print(
        f"  Pass {i}: tasks_per_env={n_tasks}, logp={out['logp'].mean():.3f}, entropy={out['entropy'].mean():.3f}"
    )

# 2. Check what reward environment gives
print("\n=== 2. Environment Reward ===")
obs = env.reset()
assert obs is not None
total_r = torch.zeros(env.B, device=dev)
steps = 0
while True:
    assert obs is not None
    with torch.no_grad():
        out = model.act(obs, temperature=10.0)
    nobs, r, done = env.step(out["new_starts"])
    total_r += r
    steps += 1
    obs = nobs if not done else None
    if done:
        break
print(f"  Total return: {total_r.mean():.2f}, steps={steps}")

# 3. Check reward breakdown
print("\n=== 3. Reward Breakdown ===")
env2 = ONTSEnv(B=4, J=9, T=100, device=dev)
obs = env2.reset()
assert obs is not None
t = 0
while True:
    assert obs is not None
    with torch.no_grad():
        out = model.act(obs, temperature=10.0)
    nobs, r, done = env2.step(out["new_starts"])
    if t < 5:
        active = env2.active.float().sum(1).mean().item()
        starts = env2.starts.sum(1).tolist()
        print(f"  t={t}: r={r.mean():.2f}, active={active:.1f}, starts={starts}")
    obs = nobs if not done else None
    t += 1
    if done:
        break

# 4. Check GAE computation
print("\n=== 4. Advantage Computation ===")
model.train()
ppo = PPO(model, device=dev, epochs=2, collect_per_update=2, lr=3e-4)

test_adv = [torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0], device=dev)]
mean, std = ppo._adv_stats(test_adv)
print(f"  Test adv stats: mean={mean:.4f}, std={std:.4f}")
print(f"  Expected: mean=12.0, std=1.58")

# 5. Collect one episode and inspect trajectory
print("\n=== 5. Trajectory Inspection ===")
obs = env.reset()
assert obs is not None
traj_steps = []
while True:
    assert obs is not None
    with torch.no_grad():
        out = model.act(obs)
    nobs, r, done = env.step(out["new_starts"])
    mask_feas = (
        obs["mask"].sum(1).tolist()
        if isinstance(obs, dict) and "mask" in obs
        else "N/A"
    )
    traj_steps.append(
        {
            "env_t": env.t - 1,
            "picks": out["new_starts"].sum(1).tolist(),
            "reward": r.tolist(),
            "value": out["value"].mean().item(),
            "logp": out["logp"].mean().item(),
            "mask_feas": mask_feas,
        }
    )
    obs = nobs if not done else None
    if done:
        break

print(f"  Traj length: {len(traj_steps)}")
for s in traj_steps[:5]:
    print(
        f"  t={s['env_t']}: picks={s['picks']}, r={s['reward'][0]:.2f}, v={s['value']:.3f}, mask_feas={s['mask_feas']}"
    )
if len(traj_steps) > 5:
    print(f"  ... ({len(traj_steps)-5} more steps)")

print("\n=== Done ===")
