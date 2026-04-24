# bohb_tpe_improved.py
# =============================================================================
# BOHB + Improved TPE (SOTA-leaning baseline, split into bohb.py + tpe.py,
# no external deps beyond numpy + scipy)
#
# Key upgrades vs your version:
# - Correct TPE candidate generation: sample FROM l(x) (good density), rank by l/g
# - Candidate pool size always >= requested n (fixes Hyperband bracket sizing)
# - Robust bandwidth (Silverman/Scott hybrid with IQR fallback + range-based floor)
# - Bounded-domain handling (reflect + clamp)
# - Better discrete handling:
#     * int: discrete kernel on integer support (mixture over observed ints + prior)
#     * choice: smoothed categorical (as before)
# - Multi-fidelity weighting: optional (budget/max_budget)^p with p>1 + normalization
# - Per-parameter prior mixture for KDE (uniform/log-uniform mixed with empirical KDE)
# - Loss split uses budget correction so low-fidelity results don't dominate
# - Optional joint modeling for conditional params (tree-structured spaces)
# - Stable config hashing (sha1 over canonical JSON)
# - More defensive behavior for tiny samples
#
# Notes:
# - This is still "independence across params" TPE (standard in practice).
# - You can plug your real objective (config,budget)->loss (lower better).
# =============================================================================

from __future__ import annotations

import concurrent.futures
import hashlib
import inspect
import json
import math
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from .pruning import PruningConfig
from .tpe import TPE, TPEConf
from .trial import Trial, TrialPruned
from .utils import _canonical_config_key


# -----------------------------------------------------------------------------
# BOHB (Hyperband + TPE proposer)
# -----------------------------------------------------------------------------


class BOHB:
    def __init__(
        self,
        config_space: dict[str, tuple],
        evaluate_fn: Callable[
            ..., float
        ],  # Can be (config, budget) or (config, budget, trial)
        min_budget: float = 1.0,
        max_budget: float = 81.0,
        eta: int = 3,
        n_iterations: int = 10,
        verbose: bool = True,
        handle_errors: bool = True,
        top_n_percent: int = 15,
        prior_trials_jsonl: str | None = None,
        pruning_mode: str = "conservative",
        history_export_jsonl: str | None = None,
        early_prune: bool = True,
        seed: int | None = None,
        tpe_conf: TPEConf | None = None,
        tpe_overrides: dict[str, Any] | None = None,
        pruning_conf: PruningConfig | None = None,
        pruning_overrides: dict[str, Any] | None = None,
        parallel_jobs: int = 1,
    ):
        self.config_space = config_space
        self.evaluate_fn = evaluate_fn
        self.min_budget = float(min_budget)
        self.max_budget = float(max_budget)
        self.eta = int(eta)
        self.n_iterations = int(n_iterations)
        self.verbose = bool(verbose)
        self.handle_errors = bool(handle_errors)
        self.top_n_percent = int(top_n_percent)
        self.pruning_mode = str(pruning_mode or "conservative").lower()
        self.history_export_jsonl = history_export_jsonl
        self.early_prune = bool(early_prune)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.parallel_jobs = int(parallel_jobs)
        self._evaluate_fn_accepts_trial = self._accepts_trial_argument(evaluate_fn)
        self.pruning_conf = (
            PruningConfig() if pruning_conf is None else pruning_conf
        ).copy_with(**dict(pruning_overrides or {}))

        conf = TPEConf() if tpe_conf is None else tpe_conf
        conf_kwargs = conf.to_kwargs()
        user_overrides = dict(tpe_overrides or {})
        overrides = {
            "max_budget": self.max_budget,
            "seed": seed,
        }
        # Preserve the previous BOHB default for startup trials unless user set it.
        if (
            "n_startup_trials" not in conf_kwargs
            and "n_startup_trials" not in user_overrides
        ):
            overrides["n_startup_trials"] = len(config_space) + 1
        overrides.update(user_overrides)
        self.tpe = TPE.from_config(config_space=config_space, cfg=conf, **overrides)

        self.history: list[dict[str, Any]] = []
        self.best_loss = float("inf")
        self.best_config: dict[str, Any] | None = None
        self.best_configs: list[tuple[dict[str, Any], float]] = []
        self._step_history: list[dict[str, float]] = []
        self._step_history_lock = threading.Lock()

        # cache: (sha1(config), budget) -> loss
        self.config_cache: dict[tuple[str, float], float] = {}

        if prior_trials_jsonl:
            self._load_prior_trials_jsonl(prior_trials_jsonl)

    def run(self) -> tuple[dict[str, Any], float]:
        s_max = int(math.log(self.max_budget / self.min_budget, self.eta))
        B = (s_max + 1) * self.max_budget

        for it in range(self.n_iterations):
            if self.verbose:
                print("\n" + "=" * 70)
                print(f"BOHB Iteration {it + 1}/{self.n_iterations}")
                print("=" * 70)

            for s in reversed(range(s_max + 1)):
                n = int(math.ceil((B / self.max_budget) * (self.eta**s) / (s + 1)))
                r = self.max_budget * (self.eta ** (-s))

                if self.verbose:
                    print(f"\nBracket s={s}: n={n}, r={r:.4g}")

                # ask for larger pool at budget r, then take top n from ranked list
                recent_bracket_success = self._recent_bracket_improvement(s)
                pool_multiplier = 4 + 8 * (1.0 - recent_bracket_success)
                pool_n = max(n * int(pool_multiplier), 32, n * 3)
                configs, scores = self.tpe.suggest(
                    n_candidates=pool_n, budget=r, return_scores="aligned"
                )
                if r >= 0.4 * self.max_budget and self.best_config:
                    n_jitter = 1 if r >= 0.7 * self.max_budget else 2
                    for _ in range(n_jitter):
                        jitter_cfg = self._jitter_config(
                            self.best_config,
                            scale=0.08 * (1.0 - r / self.max_budget),
                        )
                        if self.tpe._hard_constraints_satisfied(jitter_cfg):
                            configs.append(jitter_cfg)
                    configs = list(
                        {_canonical_config_key(c): c for c in configs}.values()
                    )
                if len(configs) > n:
                    scored = [
                        (cfg, float(scores[i]) if i < len(scores) else 0.0)
                        for i, cfg in enumerate(configs)
                    ]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    configs = [cfg for cfg, _ in scored[:n]]
                else:
                    configs = configs[:n]

                # Successive Halving within this bracket
                for i in range(s + 1):
                    n_i = max(1, int(n * (self.eta ** (-i))))
                    r_i = r * (self.eta**i)

                    if self.verbose:
                        print(f"  Round {i}/{s}: evaluating {n_i} @ budget {r_i:.4g}")

                    results = self._evaluate_configs(configs[:n_i], r_i, it, s, i)
                    for cfg, loss in results:
                        self.tpe.observe(cfg, loss, budget=r_i)
                        self._update_best(cfg, loss)

                    if not results:
                        if self.verbose:
                            print("    No valid results in this round.")
                        break

                    results_sorted = sorted(
                        results, key=lambda x: self._predict_final_loss(x[1], r_i)
                    )
                    self._update_top_configs(results_sorted)

                    # keep top 1/eta (rank by predicted final loss)
                    n_keep = max(1, int(n_i / self.eta))
                    configs = [cfg for (cfg, _) in results_sorted[:n_keep]]

        if self.best_config is None:
            raise RuntimeError("BOHB finished without any valid evaluation.")

        if self.history_export_jsonl:
            self.save_history_jsonl(self.history_export_jsonl)

        if self.verbose:
            print("\n" + "=" * 70)
            print("Optimization complete")
            print(f"Best loss : {self.best_loss:.6g}")
            print(f"Best cfg  : {self.best_config}")
            diag = self.tpe.diagnostics()
            if diag.get("trust_region_enabled", False):
                print(
                    "TR stats  : "
                    f"restarts={diag.get('trust_region_restart_count', 0)} "
                    f"length={diag.get('trust_region_length', 0.0):.4f} "
                    f"succ={diag.get('trust_region_success_count', 0)} "
                    f"fail={diag.get('trust_region_failure_count', 0)}"
                )
            print("=" * 70)

        return self.best_config, self.best_loss

    @staticmethod
    def _accepts_trial_argument(evaluate_fn: Callable[..., float]) -> bool:
        try:
            return len(inspect.signature(evaluate_fn).parameters) >= 3
        except (TypeError, ValueError):
            return False

    def _update_best(self, config: dict[str, Any], loss: float) -> None:
        if loss >= self.best_loss:
            return
        self.best_loss = float(loss)
        self.best_config = dict(config)
        if self.verbose:
            print(f"    New best: {self.best_loss:.6g}  cfg={self.best_config}")

    def _cache_key(self, config: dict[str, Any], budget: float) -> tuple[str, float]:
        key_json = _canonical_config_key(config)
        key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()
        return key_hash, float(budget)

    def _make_history_entry(
        self,
        config: dict[str, Any],
        budget: float,
        loss: float,
        raw_loss: float,
        constraint_violation: float,
        iteration: int,
        bracket: int,
        round_idx: int,
    ) -> dict[str, Any]:
        return {
            "config": dict(config),
            "budget": float(budget),
            "loss": float(loss),
            "raw_loss": float(raw_loss),
            "constraint_violation": float(constraint_violation),
            "iteration": int(iteration),
            "bracket": int(bracket),
            "round": int(round_idx),
        }

    def _evaluate_configs(
        self,
        configs: list[dict[str, Any]],
        budget: float,
        iteration: int,
        bracket: int,
        round_idx: int,
    ) -> list[tuple[dict[str, Any], float]]:
        indexed_results: list[tuple[int, dict[str, Any], float]] = []
        pending: list[tuple[int, dict[str, Any], tuple[str, float]]] = []

        for idx, config in enumerate(configs):
            if not self.tpe._hard_constraints_satisfied(config):
                if self.verbose:
                    print("    Hard constraint violated; skipping.")
                continue

            cache_key = self._cache_key(config, budget)
            if cache_key in self.config_cache:
                if self.verbose:
                    print("    (cache hit)")
                indexed_results.append((idx, config, self.config_cache[cache_key]))
                continue

            pending.append((idx, config, cache_key))

        if self.parallel_jobs > 1 and len(pending) > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.parallel_jobs
            ) as executor:
                future_to_payload = {
                    executor.submit(
                        self._evaluate_objective,
                        config,
                        budget,
                    ): (idx, config, cache_key)
                    for idx, config, cache_key in pending
                }
                completed: list[
                    tuple[
                        int,
                        dict[str, Any],
                        tuple[float, float, float] | None,
                        tuple[str, float],
                    ]
                ] = []
                for future in concurrent.futures.as_completed(future_to_payload):
                    idx, config, cache_key = future_to_payload[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        if self.handle_errors:
                            if self.verbose:
                                print(f"    Eval error in parallel job: {exc}")
                            outcome = None
                        else:
                            raise
                    completed.append((idx, config, outcome, cache_key))
        else:
            completed = []
            for idx, config, cache_key in pending:
                outcome = self._evaluate_objective(config, budget)
                completed.append((idx, config, outcome, cache_key))

        completed.sort(key=lambda item: item[0])
        for idx, config, outcome, cache_key in completed:
            loss = self._finalize_evaluation(
                config=config,
                budget=budget,
                outcome=outcome,
                cache_key=cache_key,
                iteration=iteration,
                bracket=bracket,
                round_idx=round_idx,
            )
            if loss is None:
                continue
            indexed_results.append((idx, config, loss))

        indexed_results.sort(key=lambda item: item[0])
        return [(config, loss) for _, config, loss in indexed_results]

    def _evaluate_objective(
        self,
        config: dict[str, Any],
        budget: float,
    ) -> tuple[float, float, float] | None:
        trial = Trial(config=config, budget=budget, bohb_instance=self)

        try:
            if self._evaluate_fn_accepts_trial:
                raw_loss = self.evaluate_fn(config, float(budget), trial)
            else:
                raw_loss = self.evaluate_fn(config, float(budget))
            loss = raw_loss
            if not isinstance(loss, (int, float)) or np.isnan(loss) or np.isinf(loss):
                raise ValueError(f"Invalid loss: {loss}")

            penalty = 0.0
            if self.tpe.soft_constraints:
                penalty = self.tpe._soft_constraint_violation(config)
                if penalty > 0:
                    loss = float(loss) + float(self.tpe.soft_penalty_weight) * penalty
            loss = float(loss)
            return loss, float(raw_loss), float(penalty)
        except TrialPruned:
            if self.verbose:
                print("    Trial Pruned (Intermediate step gracefully terminated).")
            return None
        except Exception as e:
            if self.handle_errors:
                if self.verbose:
                    print(f"    Eval error: {e}")
                return None
            raise

    def _finalize_evaluation(
        self,
        config: dict[str, Any],
        budget: float,
        outcome: tuple[float, float, float] | None,
        cache_key: tuple[str, float],
        iteration: int,
        bracket: int,
        round_idx: int,
    ) -> float | None:
        if outcome is None:
            return None

        loss, raw_loss, penalty = outcome
        if self.early_prune and self._should_prune(loss, budget):
            if self.verbose:
                print("    Early prune (loss above historical quantile).")
            return None

        self.config_cache[cache_key] = loss
        self.history.append(
            self._make_history_entry(
                config=config,
                budget=budget,
                loss=loss,
                raw_loss=raw_loss,
                constraint_violation=penalty,
                iteration=iteration,
                bracket=bracket,
                round_idx=round_idx,
            )
        )
        return loss

    def _update_top_configs(self, results: list[tuple[dict[str, Any], float]]) -> None:
        for cfg, loss in results:
            self.best_configs.append((dict(cfg), float(loss)))
        self.best_configs.sort(key=lambda x: x[1])
        n_keep = max(10, int(len(self.best_configs) * self.top_n_percent / 100))
        self.best_configs = self.best_configs[:n_keep]

    def _load_prior_trials_jsonl(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    cfg = rec.get("config")
                    loss = rec.get("loss")
                    budget = rec.get("budget")
                    if not isinstance(cfg, dict):
                        continue
                    if not isinstance(loss, (int, float)):
                        continue
                    if not isinstance(budget, (int, float)):
                        continue
                    loss = float(loss)
                    budget = float(budget)
                    if np.isnan(loss) or np.isinf(loss) or budget <= 0:
                        continue

                    self.tpe.observe(cfg, loss, budget=budget)
                    self.config_cache[self._cache_key(cfg, budget)] = loss
                    self.history.append(
                        self._make_history_entry(
                            config=cfg,
                            budget=budget,
                            loss=loss,
                            raw_loss=loss,
                            constraint_violation=0.0,
                            iteration=-1,
                            bracket=-1,
                            round_idx=-1,
                        )
                    )
                    if loss < self.best_loss:
                        self.best_loss = float(loss)
                        self.best_config = dict(cfg)
                    self.best_configs.append((dict(cfg), float(loss)))

            if self.best_configs:
                self.best_configs.sort(key=lambda x: x[1])
                n_keep = max(10, int(len(self.best_configs) * self.top_n_percent / 100))
                self.best_configs = self.best_configs[:n_keep]
        except FileNotFoundError:
            if self.verbose:
                print(f"Prior trials file not found: {path}")

    def _estimate_learning_curve_slope(self) -> float:
        if self.max_budget <= 0:
            return 0.0
        obs = [
            (h["budget"], h["loss"])
            for h in self.history
            if h.get("budget", 0) > 0 and h.get("loss", 0) > 0
        ]
        if len(obs) < 20:
            return 0.0
        budgets = np.array([b for b, _ in obs], dtype=float)
        losses = np.array([l for _, l in obs], dtype=float)
        if len(np.unique(budgets)) < 3:
            return 0.0
        x = np.log(budgets)
        y = np.log(losses)
        try:
            k, _ = np.polyfit(x, y, deg=1)
        except Exception:
            return 0.0
        if not np.isfinite(k):
            return 0.0
        # Conservative: limit magnitude of slope
        k = float(max(min(k, 0.0), -0.3))
        if abs(k) < 0.02:
            return 0.0
        return k

    def _predict_final_loss(self, loss: float, budget: float) -> float:
        if budget <= 0 or self.max_budget <= 0:
            return float(loss)
        if self.pruning_mode not in {"conservative", "balanced", "aggressive"}:
            return float(loss)
        k = self._estimate_learning_curve_slope()
        if k == 0.0:
            return float(loss)
        if self.pruning_mode == "balanced":
            k = k * 1.5
        elif self.pruning_mode == "aggressive":
            k = k * 2.0
        k = float(max(min(k, 0.0), -0.7))
        scale = (self.max_budget / float(budget)) ** k
        return float(loss * scale)

    def _recent_bracket_improvement(self, bracket_s: int) -> float:
        recent = [h for h in self.history[-30:] if h.get("bracket") == int(bracket_s)]
        if len(recent) < 5:
            return 0.5
        losses = [h["loss"] for h in recent]
        median = float(np.median(losses)) + 1e-8
        score = 1.0 - (min(losses) / median)
        return float(min(max(score, 0.0), 1.0))

    def _should_prune(self, current_loss: float, current_budget: float) -> bool:
        cfg = self.pruning_conf
        if (
            not self.early_prune
            or current_budget <= self.min_budget * cfg.final_budget_gate_multiplier
        ):
            return False

        relevant = [
            h["loss"]
            for h in self.history
            if h.get("budget", 0) <= current_budget
            and h.get("loss", float("inf")) < float("inf")
        ]
        if len(relevant) < cfg.final_min_history:
            return False

        progress = current_budget / self.max_budget
        q = cfg.final_quantile_base + cfg.final_quantile_growth * progress
        q = min(max(q, cfg.final_quantile_min), cfg.final_quantile_max)

        threshold = float(np.quantile(relevant, q))
        if current_loss <= threshold:
            return False

        relative_worst = (current_loss - threshold) / (np.std(relevant) + 1e-6)
        if self.pruning_mode == "aggressive":
            p_base = cfg.final_prob_base_aggressive
        elif self.pruning_mode == "balanced":
            p_base = cfg.final_prob_base_balanced
        else:
            p_base = cfg.final_prob_base_conservative
        p = min(
            cfg.final_prob_cap,
            p_base
            + cfg.final_prob_growth
            * max(0.0, relative_worst - cfg.final_relative_worst_offset),
        )

        roll = self._rng.random()
        if self.verbose and roll < p:
            print(
                f"  → Early prune triggered (loss={current_loss:.6g} > quantile {q:.2f}={threshold:.6g}, p={p:.2f})"
            )
        return roll < p

    def _should_prune_step(self, trial: Trial, step: int, loss: float) -> bool:
        """
        Intermediate Trial reporting logic:
        Decide if a Trial should be terminated at this specific step.
        """
        if not self.early_prune:
            return False

        cfg = self.pruning_conf
        progress = self._step_progress(step, trial.budget)
        if progress < cfg.step_min_progress:
            return False

        peer_losses = self._get_step_cohort_losses(step, trial.budget)
        if len(peer_losses) < cfg.step_min_history:
            return False

        peers = np.asarray(peer_losses, dtype=float)
        median = float(np.median(peers))
        mad = float(np.median(np.abs(peers - median)))
        dispersion = max(1e-6, 1.4826 * mad, 0.5 * float(np.std(peers)))

        if self.pruning_mode == "aggressive":
            base_quantile = cfg.step_quantile_aggressive
            base_sigma = cfg.step_sigma_aggressive
        elif self.pruning_mode == "balanced":
            base_quantile = cfg.step_quantile_balanced
            base_sigma = cfg.step_sigma_balanced
        else:
            base_quantile = cfg.step_quantile_conservative
            base_sigma = cfg.step_sigma_conservative

        quantile = max(
            cfg.step_quantile_floor,
            base_quantile - cfg.step_quantile_progress_slope * progress,
        )
        sigma_threshold = max(
            cfg.step_sigma_floor,
            base_sigma - cfg.step_sigma_progress_slope * progress,
        )
        quantile_threshold = float(np.quantile(peers, quantile))
        robust_threshold = median + sigma_threshold * dispersion
        threshold = max(quantile_threshold, robust_threshold)

        should_prune = float(loss) > threshold
        if should_prune and self.verbose:
            print(
                "    Intermediate prune triggered "
                f"(step={int(step)}, progress={progress:.2f}, "
                f"loss={float(loss):.6g}, threshold={threshold:.6g}, "
                f"peers={len(peer_losses)})"
            )
        return should_prune

    def _step_progress(self, step: int, budget: float) -> float:
        denom = max(float(budget), float(step + 1), 1.0)
        return min(1.0, float(step + 1) / denom)

    def _record_trial_report(self, trial: Trial, step: int, loss: float) -> None:
        record = {
            "step": float(step),
            "budget": float(trial.budget),
            "progress": self._step_progress(step, trial.budget),
            "loss": float(loss),
        }
        with self._step_history_lock:
            self._step_history.append(record)

    def _get_step_cohort_losses(self, step: int, budget: float) -> list[float]:
        cfg = self.pruning_conf
        budget = float(budget)
        target_progress = self._step_progress(step, budget)
        budget_ratio_limit = (
            float(max(self.eta, 2))
            if cfg.step_budget_ratio_limit is None
            else float(cfg.step_budget_ratio_limit)
        )

        def _budget_is_compatible(other_budget: float) -> bool:
            lo = max(min(budget, other_budget), 1e-12)
            hi = max(budget, other_budget)
            return (hi / lo) <= budget_ratio_limit

        with self._step_history_lock:
            exact_matches = [
                rec["loss"]
                for rec in self._step_history
                if int(rec["step"]) == int(step)
                and _budget_is_compatible(float(rec["budget"]))
            ]
            if len(exact_matches) >= cfg.step_exact_match_min:
                return exact_matches

            nearby_matches = [
                rec["loss"]
                for rec in self._step_history
                if abs(float(rec["progress"]) - target_progress)
                <= cfg.step_progress_tolerance
                and _budget_is_compatible(float(rec["budget"]))
            ]

        if len(nearby_matches) >= len(exact_matches):
            return nearby_matches
        return exact_matches

    def get_optimization_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    def get_top_configs(self, n: int = 10) -> list[tuple[dict[str, Any], float]]:
        return list(self.best_configs[: int(n)])

    def _jitter_config(
        self, config: dict[str, Any], scale: float = 0.08
    ) -> dict[str, Any]:
        jittered = dict(config)
        for param, spec in self.config_space.items():
            typ, rng = spec[0], spec[1]
            if param not in config:
                continue
            if typ == "float":
                lo, hi = float(rng[0]), float(rng[1])
                if isinstance(rng, tuple) and len(rng) == 3 and rng[2] == "log":
                    lo_log, hi_log = math.log10(lo), math.log10(hi)
                    val_log = math.log10(float(config[param]))
                    new_val = val_log + self._rng.normal(
                        0,
                        scale * (hi_log - lo_log),
                    )
                    jittered[param] = 10 ** float(np.clip(new_val, lo_log, hi_log))
                else:
                    val = float(config[param])
                    new_val = val + self._rng.normal(0, scale * (hi - lo))
                    jittered[param] = float(np.clip(new_val, lo, hi))
            elif typ == "int":
                lo, hi = int(rng[0]), int(rng[1])
                val = int(config[param])
                delta = max(1, int(scale * (hi - lo)))
                new_val = self._rng.integers(
                    max(lo, val - delta), min(hi + 1, val + delta + 1)
                )
                jittered[param] = int(new_val)
        return jittered

    def save_history_jsonl(self, path: str) -> None:
        """
        Save evaluation history to JSONL for transfer learning.
        Each line includes: config, budget, loss, iteration, bracket, round.
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                for h in self.history:
                    rec = {
                        "config": h.get("config", {}),
                        "budget": h.get("budget", None),
                        "loss": h.get("loss", None),
                        "iteration": h.get("iteration", None),
                        "bracket": h.get("bracket", None),
                        "round": h.get("round", None),
                    }
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
        except Exception as e:
            if self.verbose:
                print(f"Failed to export history JSONL: {e}")
