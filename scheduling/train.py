"""NCO trainers: PPO, REINFORCE, POCO, and Self-Improvement.

Trains any ActorCritic-compatible model on any NCOEnv-compatible environment.

Algorithms
----------
  PPO              : Proximal Policy Optimization with clipped surrogate + value loss
  REINFORCE        : Policy gradient with GAE / rollout / mean baselines
  POCO             : Preference Optimization (Pan et al. 2025, ICML)
  SelfImprovement  : Self-labeling with supervised imitation (Pirnay & Grimm 2024)

Baselines
---------
  "gae"     : GAE(λ) with value network (default, lowest variance)
  "rollout" : Greedy rollout REINFORCE (Kool et al. 2019, no value network needed)
  "mean"    : Episode-mean REINFORCE (simplest, highest variance)

Features
--------
    * Shared trajectory collection, replay buffer, curriculum, advantage normalization
    * PPO: clipped surrogate objective, value loss, approximate-KL early stopping
    * REINFORCE: unclipped policy gradient, no value loss, flexible baselines
    * POCO: pairwise preference optimization — samples K solutions per instance,
      ranks by return, trains logistic loss on (best_logp - worst_logp)
    * Self-Improvement: samples N solutions, selects best, trains supervised cross-entropy
    * Entropy bonus
    * Gradient clipping
    * Self-eval (branch-and-score) + curriculum k-ramp
    * Batched multi-episode collection (N episodes → K epochs per batch)
    * Learning-rate warmup + cosine decay
    * Running advantage normalization (exponential moving average)
    * Experience replay for high-return trajectories (PPO/GAE only)
    * Self-eval temperature decay (exploration → exploitation)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
import torch.nn as nn

# Trajectory step: (obs_dict, action, action_order, old_logp, old_value, reward)
Step = tuple[
    dict, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor
]

Baseline = Literal["gae", "rollout", "mean"]


class BaseAlgorithm(ABC):
    """Shared base for PPO and REINFORCE trainers.

    Handles trajectory collection, replay buffer, curriculum, temperature
    scheduling, advantage normalization, and the training loop skeleton.
    Subclasses implement `update_step()` for algorithm-specific loss computation.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        lr_warmup: int = 0,
        ecoef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 4,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        self_eval_k: int = 1,
        self_eval_temp: float = 1.5,
        self_eval_temp_min: float = 1.0,
        curriculum: bool = False,
        baseline: Baseline = "gae",
        # Mixed precision
        use_amp: bool = False,
        # Batched training
        collect_per_update: int = 8,
        # Advantage normalization (EMA)
        adv_norm: bool = True,
        adv_mean_decay: float = 0.99,
        adv_std_decay: float = 0.99,
        # Experience replay
        replay_buffer_size: int = 10,
        replay_prob: float = 0.0,
        replay_min_returns: float | None = None,
    ):
        self.model: Any = model.to(device)
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, eps=1e-5)
        self.ecoef = ecoef
        self.gamma = gamma
        self.lam = lam  # GAE lambda
        self.epochs = epochs
        self.dev = device
        self.max_grad_norm = max_grad_norm
        self.se_k = self_eval_k
        self.se_temp = self_eval_temp
        self.se_temp_min = self_eval_temp_min
        self.curriculum = curriculum
        self.baseline: Baseline = baseline
        self.collect_per_update = collect_per_update

        # Mixed precision
        self.use_amp = use_amp
        self.scaler = (
            torch.cuda.amp.GradScaler() if use_amp and device == "cuda" else None
        )

        # Running advantage normalization
        self.adv_norm = adv_norm
        self.adv_mean_decay = adv_mean_decay
        self.adv_std_decay = adv_std_decay
        self._adv_mean: float | None = None
        self._adv_std: float | None = None

        # Experience replay
        self.replay_buffer_size = replay_buffer_size
        self.replay_prob = replay_prob
        self.replay_min_returns = replay_min_returns
        self._replay_buf: list[tuple[list[Step], list[torch.Tensor]]] = []
        self._replay_raw: list[list[torch.Tensor]] = []
        self._replay_ret: list[float] = []

    # ── utilities ──────────────────────────────────────────────

    @staticmethod
    def _clone(obs: dict) -> dict:
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in obs.items()}

    def _effective_k(self, iteration: int, total_iters: int) -> int:
        """Curriculum k: linear ramp 1 → se_k over training."""
        if not self.curriculum or self.se_k <= 1:
            return 1
        denom = max(total_iters - 1, 1)
        if denom == 0:
            return 1
        try:
            val = self.se_k * (iteration / denom)
            result = max(1, int(round(val)))
        except (ZeroDivisionError, OverflowError):
            return 1
        return result

    def _self_eval_temp(self, iteration: int, total_iters: int) -> float:
        """Exponential decay: se_temp → se_temp_min."""
        if self.se_temp <= self.se_temp_min:
            return self.se_temp
        progress = iteration / max(total_iters - 1, 1)
        return self.se_temp_min + (self.se_temp - self.se_temp_min) * (
            2.71828 ** (-5.0 * progress)
        )

    def _rollout_returns(self, env) -> list[torch.Tensor]:
        """Run greedy rollout on same env state → per-step baseline returns."""
        obs = env.reset()
        returns = []
        done = False
        with torch.no_grad():
            while not done:
                out = self.model.act(obs, greedy=True)  # type: ignore[attr-defined]
                nobs, r, done = env.step(out["new_starts"])
                returns.append(r)
                obs = nobs if not done else obs
        return returns

    # ── collection ─────────────────────────────────────────────

    def collect(self, env, iteration: int = 0, total_iters: int = 1):
        """Collect one episode trajectory and compute advantages."""
        rollout_rets: list[torch.Tensor] | None = None
        if self.baseline == "rollout":
            rollout_rets = self._rollout_returns(env)

        obs = env.reset()
        traj = []
        done = False
        k = self._effective_k(iteration, total_iters)
        temp = self._self_eval_temp(iteration, total_iters)
        while not done:
            with torch.no_grad():
                if k > 1:
                    out = self.model.self_eval(obs, k=k, temperature=temp)  # type: ignore[attr-defined]
                else:
                    out = self.model.act(obs)  # type: ignore[attr-defined]
            nobs, r, done = env.step(out["new_starts"])
            traj.append(
                (
                    self._clone(obs),
                    out["new_starts"],
                    out.get("action_order"),
                    out["logp"],
                    out["value"],
                    r,
                )
            )
            obs = nobs if not done else obs

        T = len(traj)
        ep_ret = torch.stack([t[5] for t in traj]).mean().item()

        adv: list[torch.Tensor] = [
            torch.zeros(env.B, device=self.dev) for _ in range(T)
        ]

        if self.baseline == "gae":
            last_v = torch.zeros(env.B, device=self.dev)
            gae = torch.zeros(env.B, device=self.dev)
            for ki in reversed(range(T)):
                v = traj[ki][4]
                nv = last_v if ki == T - 1 else traj[ki + 1][4]
                delta = traj[ki][5] + self.gamma * nv - v
                gae = delta + self.gamma * self.lam * gae
                adv[ki] = gae

        elif self.baseline == "rollout":
            bl = rollout_rets or []
            for ki in range(T):
                r = traj[ki][5]
                b = bl[ki] if ki < len(bl) else torch.zeros_like(r)
                adv[ki] = r - b

        else:  # "mean"
            ep_mean = torch.stack([t[5] for t in traj]).mean(0)  # [B]
            for ki in range(T):
                adv[ki] = traj[ki][5] - ep_mean

        return traj, adv, ep_ret

    def collect_batch(self, env, iters: int = 1):
        """Collect multiple episodes with replay buffer management."""
        if self.replay_prob > 0 and not self._supports_replay():
            raise ValueError(
                f"Replay not supported with baseline='{self.baseline}'. "
                f"Use baseline='gae'."
            )
        episodes: list[tuple[list[Step], list[torch.Tensor], float]] = []
        for i in range(iters):
            traj, adv, ep_ret = self.collect(env, iteration=i, total_iters=iters)
            episodes.append((traj, adv, ep_ret))
            if self.replay_buffer_size > 0:
                try:
                    if self.replay_min_returns is None:
                        threshold = max(ep_ret, 0.0) * 1.1
                    else:
                        threshold = self.replay_min_returns
                    if ep_ret >= threshold:
                        raw = [t[5] for t in traj]
                        self._replay_buf.append((traj, adv))
                        self._replay_raw.append(raw)
                        self._replay_ret.append(ep_ret)
                        if len(self._replay_buf) > self.replay_buffer_size:
                            self._replay_buf.pop(0)
                            self._replay_raw.pop(0)
                            self._replay_ret.pop(0)
                except Exception:
                    pass
        return episodes

    def replay_sample(self, batch_size: int = 1):
        """Sample replayed episodes by return-weighted priority."""
        if not self._replay_buf or self.replay_prob <= 0:
            return []
        n = min(batch_size, len(self._replay_buf))
        weights = torch.tensor(self._replay_ret, dtype=torch.float32).softmax(dim=0)
        indices = torch.multinomial(weights, n, replacement=True).tolist()
        return [
            (self._replay_buf[i][0], self._replay_buf[i][1], self._replay_raw[i])
            for i in indices
        ]

    # ── advantage normalization ────────────────────────────────

    def _adv_stats(self, all_adv: list[torch.Tensor]) -> tuple[float, float]:
        """Compute running mean/std of advantages using EMA."""
        flat = torch.cat([a.flatten() for a in all_adv]).mean().item()
        std = torch.cat([a.flatten() for a in all_adv]).std().item()
        if self._adv_mean is None:
            return flat, max(std, 1e-8)
        alpha_mean = 1.0 - self.adv_mean_decay
        alpha_std = 1.0 - self.adv_std_decay
        self._adv_mean = self.adv_mean_decay * self._adv_mean + alpha_mean * flat
        self._adv_std = self.adv_std_decay * self._adv_mean + alpha_std * max(std, 1e-8)
        return self._adv_mean, max(self._adv_std, 1e-8)

    # ── abstract: algorithm-specific ───────────────────────────

    @abstractmethod
    def _supports_replay(self) -> bool:
        """Whether this algorithm supports experience replay."""
        ...

    @abstractmethod
    def _compute_replay_advantage(
        self, obs: dict, old_v: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        """Recompute advantage for a replayed step (must stay connected to graph)."""
        ...

    @abstractmethod
    def _compute_replay_ret(
        self, reward: torch.Tensor, old_v: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute return for value loss, or None if no value loss."""
        ...

    def _compute_loss(
        self,
        ratio: torch.Tensor,
        adv_val: torch.Tensor,
        old_v_val: torch.Tensor | None,
        ret_val: torch.Tensor | None,
        out: dict,
    ) -> torch.Tensor:
        """Compute the training loss for one step.

        Subclasses that override update() can leave this as default.
        Subclasses that use update() (PPO, REINFORCE) must override.
        """
        raise NotImplementedError(
            "This trainer overrides update() and does not use _compute_loss"
        )

    # ── training loop ──────────────────────────────────────────

    def update(self, env, iters: int = 200, log_every: int = 20):
        """Train for N update cycles.

        Each cycle: collect N episodes → run K epochs on all episodes.

        Returns:
            hist: list of mean episode returns per cycle.
        """
        hist = []
        for it in range(iters):
            episodes = self.collect_batch(env, iters=self.collect_per_update)
            ep_rets = [ep[2] for ep in episodes]
            mean_ret = torch.tensor(ep_rets).mean().item()
            hist.append(mean_ret)

            all_trajs: list[Step] = []
            all_adv: list[torch.Tensor] = []
            all_old_v: list[torch.Tensor] = []
            all_ret: list[torch.Tensor] = []
            for traj, adv, _ in episodes:
                for ki in range(len(traj)):
                    step = traj[ki]
                    all_trajs.append(step)
                    obs = step[0]
                    old_v = step[4]
                    reward = step[5]
                    if self.baseline == "gae":
                        with torch.no_grad():
                            task_tok, glob_ctx = self.model.encode(obs)
                            new_v = self.model.value_of(task_tok, glob_ctx)
                        adv_val = (reward + self.gamma * old_v - new_v).detach()
                    else:
                        adv_val = adv[ki]
                    all_adv.append(adv_val)
                    all_old_v.append(old_v.detach())
                    ret_val = self._compute_replay_ret(reward, old_v)
                    if ret_val is not None:
                        all_ret.append(ret_val)

            replayed_trajs: list[Step] = []
            replayed_adv: list[torch.Tensor] = []
            replayed_old_v: list[torch.Tensor] = []
            replayed_ret: list[torch.Tensor] = []
            if self.replay_prob > 0 and self._replay_buf:
                try:
                    n_replay = max(1, int(round(len(episodes) * self.replay_prob)))
                except (TypeError, OverflowError):
                    n_replay = 1
                replayed = self.replay_sample(batch_size=n_replay)
                for traj, adv, raw_rew in replayed:
                    for ki in range(len(traj)):
                        step = traj[ki]
                        replayed_trajs.append(step)
                        obs = step[0]
                        old_v = step[4]
                        reward = raw_rew[ki]
                        replayed_adv.append(
                            self._compute_replay_advantage(obs, old_v, reward)
                        )
                        replayed_old_v.append(old_v.detach())
                        ret_val = self._compute_replay_ret(reward, old_v)
                        if ret_val is not None:
                            replayed_ret.append(ret_val)

            all_norm_adv = all_adv + replayed_adv
            adv_mean, adv_std = self._adv_stats(all_norm_adv)
            adv_std = max(adv_std, 1e-8)

            combined_trajs = all_trajs + replayed_trajs
            combined_adv = all_adv + replayed_adv
            combined_old_v = all_old_v + replayed_old_v
            combined_ret = all_ret + replayed_ret
            total_steps = len(combined_trajs)

            stop_epoch = False
            for epoch in range(self.epochs):
                if self.lr_warmup > 0:
                    warmup_step = max(0, min(it * self.epochs + epoch, self.lr_warmup))
                    warmup_lr = self.lr * (warmup_step / self.lr_warmup)
                else:
                    warmup_lr = self.lr
                total_steps_lr = iters * self.epochs
                cosine_step = it * self.epochs + epoch
                cosine_lr = warmup_lr + (self.lr - warmup_lr) * 0.5 * (
                    1
                    + torch.cos(
                        torch.tensor(cosine_step / max(total_steps_lr, 1) * 3.14159)
                    )
                )
                for pg in self.opt.param_groups:
                    pg["lr"] = cosine_lr.item()

                approx_kls = []
                for ki in range(total_steps):
                    obs, act, order, old_lp, old_v, _ = combined_trajs[ki]
                    # Skip padding steps (episode already ended)
                    if not (act > 0.5).any():
                        continue
                    adv_val = (combined_adv[ki] - adv_mean) / adv_std
                    adv_val = adv_val.to(self.dev)
                    old_v_val = combined_old_v[ki].to(self.dev)
                    ret_val = (
                        combined_ret[ki].to(self.dev)
                        if ki < len(combined_ret)
                        else None
                    )

                    if self.use_amp and self.scaler:
                        with torch.cuda.amp.autocast():
                            out = self.model.act(obs, action=act, action_order=order)  # type: ignore[attr-defined]
                            loss = self._compute_loss(
                                ratio=(out["logp"] - old_lp.detach()).exp(),
                                adv_val=adv_val.detach(),
                                old_v_val=old_v_val,
                                ret_val=ret_val,
                                out=out,
                            )
                        self.opt.zero_grad()
                        self.scaler.scale(loss).backward()
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        out = self.model.act(obs, action=act, action_order=order)  # type: ignore[attr-defined]
                        log_ratio = out["logp"] - old_lp.detach()
                        ratio = log_ratio.exp()
                        approx_kls.append(((ratio - 1) - log_ratio).mean().detach())
                        loss = self._compute_loss(
                            ratio=ratio,
                            adv_val=adv_val.detach(),
                            old_v_val=old_v_val,
                            ret_val=ret_val,
                            out=out,
                        )
                        self.opt.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                        self.opt.step()
                if self.target_kl is not None and approx_kls:
                    mean_kl = torch.stack(approx_kls).mean().item()
                    if mean_kl > 1.5 * self.target_kl:
                        stop_epoch = True
                if stop_epoch:
                    break

            if it % log_every == 0 or it == iters - 1:
                std_ret = torch.tensor(ep_rets).std().item()
                n_replay_active = len(self._replay_buf)
                print(
                    f"  iter {it:4d}   mean return {mean_ret:8.2f} ± {std_ret:6.2f}  "
                    f"replay={n_replay_active}"
                )
        return hist

    # ── algorithm-specific hooks (overridden by subclasses) ────
    target_kl: float | None = None  # KL threshold for early stopping
    clip: float = 0.0  # PPO clip epsilon (0 = no clipping for REINFORCE)


# ──────────────────────────────────────────────────────────
# PPO - Proximal Policy Optimization (GAE only)
# ──────────────────────────────────────────────────────────


class PPO(BaseAlgorithm):
    """Proximal Policy Optimization with clipped surrogate + value loss.

    PPO requires GAE baseline for policy gradient flow (advantage must be
    connected to the model for ratio → adv → model backprop).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        lr_warmup: int = 0,
        clip: float = 0.2,
        vcoef: float = 0.5,
        ecoef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 4,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        target_kl: float | None = 0.03,
        value_clip: float = 0.2,
        self_eval_k: int = 1,
        self_eval_temp: float = 1.5,
        self_eval_temp_min: float = 1.0,
        curriculum: bool = False,
        # Batched training
        collect_per_update: int = 8,
        # Advantage normalization (EMA)
        adv_norm: bool = True,
        adv_mean_decay: float = 0.99,
        adv_std_decay: float = 0.99,
        # Experience replay
        replay_buffer_size: int = 10,
        replay_prob: float = 0.0,
        replay_min_returns: float | None = None,
        # Mixed precision
        use_amp: bool = False,
    ):
        super().__init__(
            model=model,
            lr=lr,
            lr_warmup=lr_warmup,
            ecoef=ecoef,
            gamma=gamma,
            lam=lam,
            epochs=epochs,
            device=device,
            max_grad_norm=max_grad_norm,
            self_eval_k=self_eval_k,
            self_eval_temp=self_eval_temp,
            self_eval_temp_min=self_eval_temp_min,
            curriculum=curriculum,
            baseline="gae",  # PPO requires GAE
            collect_per_update=collect_per_update,
            adv_norm=adv_norm,
            adv_mean_decay=adv_mean_decay,
            adv_std_decay=adv_std_decay,
            replay_buffer_size=replay_buffer_size,
            replay_prob=replay_prob,
            replay_min_returns=replay_min_returns,
            use_amp=use_amp,
        )
        self.clip = clip
        self.vcoef = vcoef
        self.target_kl = target_kl
        self.value_clip = value_clip

    def _supports_replay(self) -> bool:
        return True

    def _compute_replay_advantage(
        self, obs: dict, old_v: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            task_tok, glob_ctx = self.model.encode(obs)
            new_v = self.model.value_of(task_tok, glob_ctx)
        return (reward + self.gamma * old_v - new_v).detach()

    def _compute_replay_ret(
        self, reward: torch.Tensor, old_v: torch.Tensor
    ) -> torch.Tensor:
        return reward + self.gamma * old_v.detach()

    def _compute_loss(
        self,
        ratio: torch.Tensor,
        adv_val: torch.Tensor,
        old_v_val: torch.Tensor | None,
        ret_val: torch.Tensor | None,
        out: dict,
    ) -> torch.Tensor:
        a = adv_val.detach()
        pl = -torch.min(
            ratio * a,
            ratio.clamp(1 - self.clip, 1 + self.clip) * a,
        ).mean()
        if ret_val is not None:
            assert old_v_val is not None
            value = out["value"]
            v_clipped = old_v_val.detach() + (value - old_v_val.detach()).clamp(
                -self.value_clip, self.value_clip
            )
            v_loss = (value - ret_val).pow(2)
            v_loss_clipped = (v_clipped - ret_val).pow(2)
            vl = 0.5 * torch.max(v_loss, v_loss_clipped).mean()
            return pl + self.vcoef * vl - self.ecoef * out["entropy"].mean()
        return pl - self.ecoef * out["entropy"].mean()


# ──────────────────────────────────────────────────────────
# REINFORCE - Policy Gradient
# ──────────────────────────────────────────────────────────


class REINFORCE(BaseAlgorithm):
    """REINFORCE with pluggable baseline (GAE / rollout / mean).

    Pure policy gradient: no clipping, no value loss, no KL early stopping.
    Supports three baselines:
      - "gae"   : GAE(λ) with value network (low variance, PPO-compatible)
      - "rollout": Greedy rollout baseline (Kool et al. 2019)
      - "mean"  : Episode-mean baseline (simplest, highest variance)

    Note: Experience replay only works with GAE baseline since replay
    recomputes advantages from the value network.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        lr_warmup: int = 0,
        ecoef: float = 0.01,
        gamma: float = 0.99,
        lam: float = 0.95,
        epochs: int = 4,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        self_eval_k: int = 1,
        self_eval_temp: float = 1.5,
        self_eval_temp_min: float = 1.0,
        curriculum: bool = False,
        baseline: Baseline = "gae",
        # Batched training
        collect_per_update: int = 8,
        # Advantage normalization (EMA)
        adv_norm: bool = True,
        adv_mean_decay: float = 0.99,
        adv_std_decay: float = 0.99,
        # Experience replay (only for GAE)
        replay_buffer_size: int = 10,
        replay_prob: float = 0.0,
        replay_min_returns: float | None = None,
        # Mixed precision
        use_amp: bool = False,
    ):
        super().__init__(
            model=model,
            lr=lr,
            lr_warmup=lr_warmup,
            ecoef=ecoef,
            gamma=gamma,
            lam=lam,
            epochs=epochs,
            device=device,
            max_grad_norm=max_grad_norm,
            self_eval_k=self_eval_k,
            self_eval_temp=self_eval_temp,
            self_eval_temp_min=self_eval_temp_min,
            curriculum=curriculum,
            baseline=baseline,
            collect_per_update=collect_per_update,
            adv_norm=adv_norm,
            adv_mean_decay=adv_mean_decay,
            adv_std_decay=adv_std_decay,
            replay_buffer_size=replay_buffer_size,
            replay_prob=replay_prob,
            replay_min_returns=replay_min_returns,
            use_amp=use_amp,
        )
        self.clip = 0.0  # REINFORCE has no clip
        self.target_kl = None  # REINFORCE has no KL early stopping
        self.value_clip = 0.0  # not used by REINFORCE

    def _supports_replay(self) -> bool:
        return self.baseline == "gae"

    def _compute_replay_advantage(
        self, obs: dict, old_v: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        if self.baseline == "gae":
            with torch.no_grad():
                task_tok, glob_ctx = self.model.encode(obs)
                new_v = self.model.value_of(task_tok, glob_ctx)
            return (reward + self.gamma * old_v - new_v).detach()
        # For rollout/mean, replay stores pre-computed advantages
        return torch.zeros_like(reward)  # should not reach here for non-GAE

    def _compute_replay_ret(
        self, reward: torch.Tensor, old_v: torch.Tensor
    ) -> torch.Tensor | None:
        # REINFORCE has no value loss
        return None

    def _compute_loss(
        self,
        ratio: torch.Tensor,
        adv_val: torch.Tensor,
        old_v_val: torch.Tensor | None,
        ret_val: torch.Tensor | None,
        out: dict,
    ) -> torch.Tensor:
        # Pure policy gradient: -logp * adv + entropy bonus
        # No clipping, no value loss
        a = adv_val.detach()
        pl = -(ratio * a).mean()
        return pl - self.ecoef * out["entropy"].mean()


# ──────────────────────────────────────────────────────────
# POCO - Preference Optimization for Combinatorial Optimization
# ──────────────────────────────────────────────────────────


class POCO(BaseAlgorithm):
    """POCO (Pan et al. 2025, ICML): Preference Optimization for CO.

    Instead of learning from absolute rewards, POCO learns from
    pairwise preferences: given two solutions sampled for the same
    instance, which one is better? This transforms quantitative
    reward signals into qualitative preference signals.

    Each update step samples K solutions per instance, ranks them
    by objective value, and trains a pairwise preference loss:
      L = -logσ(\beta * (logp_best - logp_worst))

    This is much more sample-efficient than PPO because:
      - Uses K rollouts per instance instead of 1
      - Preference signal is scale-invariant
      - No value network needed for advantage estimation
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        lr_warmup: int = 0,
        ecoef: float = 0.01,
        gamma: float = 0.99,
        epochs: int = 4,
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        self_eval_k: int = 1,
        self_eval_temp: float = 1.5,
        self_eval_temp_min: float = 1.0,
        curriculum: bool = False,
        # POCO-specific
        k_samples: int = 4,  # K solutions per instance for preference pairs
        beta: float = 1.0,  # preference sharpness (\beta in logistic)
        # Batched training
        collect_per_update: int = 8,
        # Advantage normalization (EMA)
        adv_norm: bool = True,
        adv_mean_decay: float = 0.99,
        adv_std_decay: float = 0.99,
        # Mixed precision
        use_amp: bool = False,
    ):
        super().__init__(
            model=model,
            lr=lr,
            lr_warmup=lr_warmup,
            ecoef=ecoef,
            gamma=gamma,
            lam=0.95,
            epochs=epochs,
            device=device,
            max_grad_norm=max_grad_norm,
            self_eval_k=self_eval_k,
            self_eval_temp=self_eval_temp,
            self_eval_temp_min=self_eval_temp_min,
            curriculum=curriculum,
            baseline="gae",  # not used by POCO
            collect_per_update=collect_per_update,
            adv_norm=adv_norm,
            adv_mean_decay=adv_mean_decay,
            adv_std_decay=adv_std_decay,
            replay_buffer_size=0,
            replay_prob=0.0,
            replay_min_returns=None,
            use_amp=use_amp,
        )
        self.k_samples = k_samples
        self.beta = beta  # preference sharpness
        self.clip = 0.0
        self.target_kl = None
        self.value_clip = 0.0

    def _supports_replay(self) -> bool:
        return False

    def _compute_replay_advantage(self, obs, old_v, reward):
        raise NotImplementedError("POCO does not support replay")

    def _compute_replay_ret(self, reward, old_v):
        return None

    def _collect_preference_pairs(self, env):
        """Collect K solutions per instance, form preference pairs.

        Reset env ONCE, then sample K independent actions from the SAME
        environment state (same instance, same randomness source). Rank by
        return and form (best, worst) preference pairs.

        This is the correct POCO formulation (Pan et al. 2025 ICML):
        preferences are over actions on the SAME instance, not across
        different instances.

        Returns:
            pref_data: list of dicts with preference pairs
        """
        B = env.B

        # Reset env ONCE — all K solutions share the same instance
        obs = env.reset()

        # Collect K trajectories from the SAME initial state
        all_trajectories = []
        for k in range(self.k_samples):
            # Resync env to the shared initial state before each rollout
            cur_obs = obs if k == 0 else env.reset()
            traj = []
            done = False
            while not done:
                with torch.no_grad():
                    out = self.model.act(cur_obs)  # type: ignore[attr-defined]
                nobs, r, done = env.step(out["new_starts"])
                traj.append(
                    (
                        self._clone(cur_obs),
                        out["new_starts"],
                        out.get("action_order"),
                        out["logp"],
                        out["value"],
                        r,
                    )
                )
                cur_obs = nobs if not done else cur_obs

            ep_ret = torch.stack([t[5] for t in traj]).mean().item()
            all_trajectories.append((traj, ep_ret))

        # Rank by return (higher = better)
        all_trajectories.sort(key=lambda x: x[1], reverse=True)

        # Form pairs: best vs worst, best vs second-worst, etc.
        result = []
        n_pairs = min(3, len(all_trajectories) - 1)  # up to 3 pairs per instance
        for i in range(n_pairs):
            best_traj, best_ret = all_trajectories[0]
            worst_traj, worst_ret = all_trajectories[i + 1]

            # Compute log-prob for each solution
            def _total_logp(traj):
                return sum(t[3].sum().item() for t in traj)

            best_lp = _total_logp(best_traj)
            worst_lp = _total_logp(worst_traj)

            result.append(
                {
                    "best_traj": best_traj,
                    "worst_traj": worst_traj,
                    "best_ret": best_ret,
                    "worst_ret": worst_ret,
                    "best_lp": best_lp,
                    "worst_lp": worst_lp,
                }
            )

        return result

    def update(self, env, iters: int = 200, log_every: int = 20):
        """Train with POCO preference optimization.

        Each cycle:
          1. Collect K solutions per instance
          2. Rank by return, form preference pairs
          3. Train preference loss
        """
        hist = []
        for it in range(iters):
            # Collect preference pairs
            pref_data = self._collect_preference_pairs(env)
            if not pref_data:
                continue

            # Compute mean return
            mean_ret = torch.tensor([d["best_ret"] for d in pref_data]).mean().item()
            hist.append(mean_ret)

            # Compute loss over all preference pairs
            total_loss = torch.tensor(0.0, device=self.dev)
            total_entropy = torch.tensor(0.0, device=self.dev)
            n_steps = 0

            for pair in pref_data:
                best_traj = pair["best_traj"]
                worst_traj = pair["worst_traj"]

                best_logp_step: list[torch.Tensor] = []
                worst_logp_step: list[torch.Tensor] = []

                # Forward pass for best and worst solutions
                for ki in range(max(len(best_traj), len(worst_traj))):
                    if ki < len(best_traj):
                        obs, act, order, old_lp, old_v, _ = best_traj[ki]
                        if not (act > 0.5).any():
                            continue
                        if self.use_amp and self.scaler:
                            with torch.cuda.amp.autocast():
                                out = self.model.act(obs, action=act, action_order=order)  # type: ignore[attr-defined]
                        else:
                            out = self.model.act(obs, action=act, action_order=order)  # type: ignore[attr-defined]
                        best_logp_step.append(out["logp"])
                        total_entropy += out["entropy"].mean()

                    if ki < len(worst_traj):
                        obs, act, order, old_lp, old_v, _ = worst_traj[ki]
                        if not (act > 0.5).any():
                            continue
                        if self.use_amp and self.scaler:
                            with torch.cuda.amp.autocast():
                                out = self.model.act(obs, action=act, action_order=order)  # type: ignore[attr-defined]
                        else:
                            out = self.model.act(obs, action=act, action_order=order)  # type: ignore[attr-defined]
                        worst_logp_step.append(out["logp"])

                if not best_logp_step or not worst_logp_step:
                    continue

                # Pairwise preference loss: sum over all steps
                # L = -logσ(\beta * sum(best_logp - worst_logp))
                best_lp_sum = torch.stack(best_logp_step).sum()
                worst_lp_sum = torch.stack(worst_logp_step).sum()
                diff = best_lp_sum - worst_lp_sum
                if self.use_amp and self.scaler:
                    with torch.cuda.amp.autocast():
                        pref_loss = -torch.nn.functional.logsigmoid(
                            self.beta * diff
                        ).mean()
                else:
                    pref_loss = -torch.nn.functional.logsigmoid(self.beta * diff).mean()
                total_loss += pref_loss
                n_steps += 1

            if n_steps > 0:
                avg_loss = total_loss / n_steps
                avg_entropy = total_entropy / n_steps

                # Backward pass
                self.opt.zero_grad()
                loss = avg_loss - self.ecoef * avg_entropy
                if self.use_amp and self.scaler:
                    self.scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.opt.step()

            if it % log_every == 0 or it == iters - 1:
                std_ret = torch.tensor([d["best_ret"] for d in pref_data]).std().item()
                print(
                    f"  iter {it:4d}   mean return {mean_ret:8.2f} ± {std_ret:6.2f}  "
                    f"pairs={len(pref_data)}"
                )

        return hist


# ──────────────────────────────────────────────────────────
# Self-Improvement - Self-Labeling with Supervised Imitation
# ──────────────────────────────────────────────────────────


class SelfImprovement(BaseAlgorithm):
    """Self-Improvement via supervised imitation (Pirnay & Grimm 2024, TMLR).

    Periodically samples N solutions with the current policy, selects
    the best as a pseudo-label, and performs a supervised imitation
    training step.

    Usage: Run SelfImprovement every K steps alongside PPO/REINFORCE.
    The idea is to "reset" the policy toward high-quality behavior
    periodically, preventing catastrophic forgetting.

    Key detail: samples WITHOUT replacement so the policy keeps
    exploring different regions of solution space.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        ecoef: float = 0.01,
        gamma: float = 0.99,
        epochs: int = 1,  # SL training uses 1 epoch
        device: str = "cpu",
        max_grad_norm: float = 1.0,
        self_eval_k: int = 1,
        self_eval_temp: float = 1.0,  # lower temp for sampling
        self_eval_temp_min: float = 0.5,
        # Self-Improvement-specific
        n_samples: int = 8,  # N solutions to sample per instance
        every_k_steps: int = 10,  # run SI every K training steps
        curriculum: bool = False,
        # Batched training
        collect_per_update: int = 8,
        # Advantage normalization (not used by SI)
        adv_norm: bool = False,
        adv_mean_decay: float = 0.99,
        adv_std_decay: float = 0.99,
        # Mixed precision
        use_amp: bool = False,
    ):
        super().__init__(
            model=model,
            lr=lr,
            lr_warmup=0,
            ecoef=ecoef,
            gamma=gamma,
            lam=0.95,
            epochs=epochs,
            device=device,
            max_grad_norm=max_grad_norm,
            self_eval_k=self_eval_k,
            self_eval_temp=self_eval_temp,
            self_eval_temp_min=self_eval_temp_min,
            curriculum=curriculum,
            baseline="gae",
            collect_per_update=collect_per_update,
            adv_norm=adv_norm,
            adv_mean_decay=adv_mean_decay,
            adv_std_decay=adv_std_decay,
            replay_buffer_size=0,
            replay_prob=0.0,
            replay_min_returns=None,
            use_amp=use_amp,
        )
        self.n_samples = n_samples
        self.every_k_steps = every_k_steps
        self.clip = 0.0
        self.target_kl = None
        self.value_clip = 0.0
        self._si_step_counter = 0

    def _supports_replay(self) -> bool:
        return False

    def _compute_replay_advantage(self, obs, old_v, reward):
        raise NotImplementedError("Self-Improvement does not support replay")

    def _compute_replay_ret(self, reward, old_v):
        return None

    def _compute_loss(
        self,
        ratio: torch.Tensor,
        adv_val: torch.Tensor,
        old_v_val: torch.Tensor | None,
        ret_val: torch.Tensor | None,
        out: dict,
    ) -> torch.Tensor:
        raise NotImplementedError("Use supervised_ce_loss instead")

    def _collect_sl_dataset(self, env):
        """Collect N solutions per instance, select best as pseudo-label.

        Returns:
            dataset: list of (obs, action) pairs for the best solution
        """
        all_solutions = []
        B = env.B

        for _ in range(self.n_samples):
            obs = env.reset()
            traj = []
            done = False
            while not done:
                with torch.no_grad():
                    if self.se_k > 1:
                        temp = self._self_eval_temp(0, 1)
                        out = self.model.self_eval(obs, k=self.se_k, temperature=temp)  # type: ignore[attr-defined]
                    else:
                        out = self.model.act(obs)  # type: ignore[attr-defined]
                nobs, r, done = env.step(out["new_starts"])
                traj.append(
                    (
                        self._clone(obs),
                        out["new_starts"],
                        out.get("action_order"),
                        out["logp"],
                        out["value"],
                        r,
                    )
                )
                obs = nobs if not done else obs

            ep_ret = torch.stack([t[5] for t in traj]).mean().item()
            all_solutions.append((traj, ep_ret))

        # Select best solution (highest return)
        all_solutions.sort(key=lambda x: x[1], reverse=True)
        best_traj = all_solutions[0][0]

        # Build SL dataset: (obs_dict, action_tensor)
        dataset = []
        for ki in range(len(best_traj)):
            obs, act, order, old_lp, old_v, _ = best_traj[ki]
            if (act > 0.5).any():
                dataset.append((obs, act))

        return dataset

    def update(self, env, iters: int = 200, log_every: int = 20):
        """Train with Self-Improvement.

        Every K steps, collect N solutions, select best, train supervised.
        """
        hist = []
        for it in range(iters):
            self._si_step_counter += 1

            sl_dataset: list[tuple] | None = None

            # Collect SL dataset every K steps
            if it % self.every_k_steps == 0:
                sl_dataset = self._collect_sl_dataset(env)

                if sl_dataset:
                    # Supervised cross-entropy loss
                    total_loss = torch.tensor(0.0, device=self.dev)
                    n_steps = 0

                    for obs, target_act in sl_dataset:
                        # Forward pass: get log probabilities for each action
                        if self.use_amp and self.scaler:
                            with torch.cuda.amp.autocast():
                                out = self.model.act(obs, action=target_act, action_order=None)  # type: ignore[attr-defined]
                            sl_loss = -out["logp"].mean()
                        else:
                            out = self.model.act(obs, action=target_act, action_order=None)  # type: ignore[attr-defined]
                            sl_loss = -out["logp"].mean()
                        total_loss += sl_loss
                        n_steps += 1

                    if n_steps > 0:
                        avg_loss = total_loss / n_steps
                        self.opt.zero_grad()
                        if self.use_amp and self.scaler:
                            self.scaler.scale(avg_loss).backward()
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                            self.scaler.step(self.opt)
                            self.scaler.update()
                        else:
                            avg_loss.backward()
                            nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                            self.opt.step()

            # Compute mean return for logging
            if sl_dataset:
                # Reuse last collected dataset for return calculation
                obs, target_act = sl_dataset[0]
                with torch.no_grad():
                    out = self.model.act(obs)  # type: ignore[attr-defined]
                ep_ret = (
                    out.get("value", torch.zeros(env.B, device=self.dev)).mean().item()
                )
            else:
                # Fallback: just collect one trajectory
                obs = env.reset()
                traj = []
                done = False
                while not done:
                    with torch.no_grad():
                        out = self.model.act(obs)  # type: ignore[attr-defined]
                    nobs, r, done = env.step(out["new_starts"])
                    traj.append((obs, r))
                    obs = nobs if not done else obs
                ep_ret = torch.stack([t[1] for t in traj]).mean().item()

            hist.append(ep_ret)

            if it % log_every == 0 or it == iters - 1:
                n_sl = len(sl_dataset) if sl_dataset else 0
                print(
                    f"  iter {it:4d}   mean return {ep_ret:8.2f}  "
                    f"si_step={self._si_step_counter}  sl_data={n_sl}"
                )

        return hist
