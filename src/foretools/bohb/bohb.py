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

from .hyperband import HyperbandScheduler
from .pruning import PruningConfig
from .tpe import TPE, TPEConf
from .trial import Trial, TrialPruned
from .utils import _canonical_config_key


class _ImmediateFuture:
    """Future-shaped wrapper around an already-computed value, used when
    parallel_jobs == 1 so the ASHA loop has a single submit/await path."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def result(self) -> Any:
        return self._value


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
        # Convergence detection / adaptive stopping
        max_no_improvement_rounds: int | None = None,
        convergence_threshold: float = 1e-6,
        min_improvement_frac: float = 0.001,
        convergence_lookback: int = 10,
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

        # Hyperband rung grid for snapping cache-key budgets so that float
        # noise (27.0 vs 27.0000001) doesn't cause cache misses.
        self._rung_budgets: list[float] = self._build_rung_grid()
        # Cached learning-curve slope: refit every _slope_refresh_every new
        # history entries, since the slope barely moves between fits.
        self._slope_cached: float = 0.0
        self._slope_history_len: int = -1
        self._slope_refresh_every: int = 10
        # Indices over _step_history for O(1) cohort lookup. Each entry is
        # (budget, progress, loss) so cohort filters can be applied without
        # scanning the full history.
        self._step_by_step: dict[int, list[tuple[float, float, float]]] = {}
        self._step_by_progress_bin: dict[int, list[tuple[float, float, float]]] = {}
        self._progress_bin_width = 0.05

        # cache: (sha1(config), budget) -> loss
        self.config_cache: dict[tuple[str, float], float] = {}

        if prior_trials_jsonl:
            self._load_prior_trials_jsonl(prior_trials_jsonl)

        # Convergence detection parameters
        self.max_no_improvement_rounds = (
            max_no_improvement_rounds
            if max_no_improvement_rounds is not None
            else max(20, int(n_iterations * 0.4))  # Default: 40% of total rounds
        )
        self.convergence_threshold = float(convergence_threshold)
        self.min_improvement_frac = float(min_improvement_frac)
        self.convergence_lookback = int(convergence_lookback)
        self._round_losses: list[float] = []  # Track best loss per round

    def check_convergence(self) -> dict[str, Any]:
        """
        Check whether optimization has converged.

        Uses multiple signals (at least 2 of 3 must agree):
        1. No-improvement count: no significant improvement in N consecutive rounds.
        2. Plateau: last K rounds show < threshold improvement and low range.
        3. Low improvement variance: recent improvements are consistently near zero.

        Returns:
            Dict with:
                - converged: bool
                - reason: str | None
                - details: dict
        """
        if len(self._round_losses) < 5:
            return {"converged": False, "reason": None, "details": {}}

        losses = self._round_losses
        n = len(losses)
        recent = losses[-self.convergence_lookback :]

        # === Signal 1: No-improvement count ===
        # Count consecutive rounds where no improvement occurred
        rounds_without_improvement = 0
        running_best = float("inf")
        for l in reversed(losses):
            if l < running_best:
                running_best = l
                rounds_without_improvement = 0  # Reset at improvement
            else:
                rounds_without_improvement += 1

        no_improvement_detected = (
            rounds_without_improvement >= self.max_no_improvement_rounds
        )

        # === Signal 2: Plateau detection (last K rounds) ===
        # Check if the last few rounds have settled
        k_plateau = min(5, len(recent))
        last_k = recent[-k_plateau:]
        best_last_k = min(last_k)
        range_last_k = max(last_k) - min(last_k)

        # Improvement from start of window to current best
        window_start = (
            losses[-self.convergence_lookback]
            if self.convergence_lookback < n
            else losses[0]
        )
        improvement_frac = (window_start - best_last_k) / (abs(window_start) + 1e-8)

        # Use a relative threshold: improvement must be tiny compared to best loss
        plateau_detected = (
            range_last_k < self.convergence_threshold * max(abs(best_last_k), 1e-6)
            and improvement_frac
            < self.min_improvement_frac * 3  # Relaxed for early rounds
        )

        # === Signal 3: Low improvement variance ===
        # Only look at the most recent 5 improvements
        k_var = min(5, len(recent) - 1)
        if k_var >= 2:
            recent_improvements = [recent[i] - recent[i + 1] for i in range(k_var)]
            improvement_var = float(np.var(recent_improvements))
            # Use relative variance: variance should be tiny compared to improvement scale
            improvement_scale = abs(best_last_k) + 1e-6
            low_var = (
                improvement_var < (self.convergence_threshold * improvement_scale) ** 2
            )
        else:
            low_var = False

        # Overall convergence: at least 2 of 3 signals
        signals = [no_improvement_detected, plateau_detected, low_var]
        n_signals = sum(signals)

        if n_signals >= 2 or (n_signals >= 1 and no_improvement_detected):
            reasons = []
            if no_improvement_detected:
                reasons.append(
                    f"no improvement for {rounds_without_improvement} rounds"
                )
            if plateau_detected:
                reasons.append(
                    f"plateau (range={range_last_k:.6g} over last {k_plateau} rounds)"
                )
            if low_var:
                reasons.append(f"low improvement variance={improvement_var:.2e}")

            return {
                "converged": True,
                "reason": ", ".join(reasons),
                "details": {
                    "no_improvement": no_improvement_detected,
                    "plateau": plateau_detected,
                    "low_var": low_var,
                    "signals_met": n_signals,
                    "rounds_without_improvement": rounds_without_improvement,
                    "current_best": best_last_k,
                },
            }

        return {
            "converged": False,
            "reason": None,
            "details": {
                "no_improvement": no_improvement_detected,
                "plateau": plateau_detected,
                "low_var": low_var,
                "signals_met": n_signals,
                "rounds_without_improvement": rounds_without_improvement,
                "current_best": best_last_k,
            },
        }

    def _predict_improvement(self) -> float:
        """
        Predict the expected improvement from the next iteration
        using learning curve extrapolation.

        Returns:
            Predicted improvement (positive = more improvement expected).
        """
        if len(self._round_losses) < 3:
            return float("inf")

        # Extrapolate: the best loss in the next round is predicted
        # to be: current_best * (1 + predicted_improvement_frac)
        # where predicted_improvement_frac comes from the learning curve slope.
        k = self._estimate_learning_curve_slope()
        if k == 0.0:
            return 0.0

        # Use the slope to predict how much lower loss will go
        # with one more budget increase
        current_best = min(self._round_losses)
        # Predicted improvement is roughly: current_best * (1 - scale)
        # where scale depends on the slope and budget ratio.
        scale = (self.max_budget / (self.max_budget * 0.5)) ** k if k < 0 else 0.0
        predicted_improvement = abs(current_best * scale)

        return float(max(predicted_improvement, 0.0))

    def run(self) -> tuple[dict[str, Any], float]:
        scheduler = HyperbandScheduler(
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            eta=self.eta,
        )

        converged_info = {"converged": False, "reason": None}
        iteration_count = 0

        for it in range(self.n_iterations):
            iteration_count += 1
            if self.verbose:
                print("\n" + "=" * 70)
                print(f"BOHB Iteration {iteration_count}/{self.n_iterations}")
                print("=" * 70)

            for s, n, r in scheduler.brackets():
                self._run_bracket(it, s, n, r, scheduler)

            # Record best loss for this round; skip inf to avoid false convergence
            if self.best_loss < float("inf"):
                self._round_losses.append(self.best_loss)

            # Check convergence after each bracket completes
            if not converged_info.get("converged", False):
                converged_info = self.check_convergence()
                if converged_info["converged"]:
                    if self.verbose:
                        print(
                            f"\n  >> Convergence detected: {converged_info['reason']}"
                        )
                        print(f"  >> Stopping early (best loss: {self.best_loss:.6g})")
                    break

        if self.best_config is None:
            raise RuntimeError("BOHB finished without any valid evaluation.")

        if self.history_export_jsonl:
            self.save_history_jsonl(self.history_export_jsonl)

        if self.verbose:
            print("\n" + "=" * 70)
            print("Optimization complete")
            print(f"Best loss : {self.best_loss:.6g}")
            print(f"Best cfg  : {self.best_config}")
            print(f"Rounds    : {iteration_count}/{self.n_iterations}")
            if converged_info.get("converged"):
                print(f"Converged : Yes — {converged_info['reason']}")
            diag = self.tpe.diagnostics()
            if diag.get("trust_region_enabled", False):
                print(
                    "TR stats  : "
                    f"restarts={diag.get('trust_region_restart_count', 0)} "
                    f"length={diag.get('trust_region_length', 0.0):.4f} "
                    f"succ={diag.get('trust_region_success_count', 0)} "
                    f"fail={diag.get('trust_region_failure_count', 0)} "
                    f"per_param={diag.get('trust_region_lengths', 'N/A')}"
                )
            print("=" * 70)

        return self.best_config, self.best_loss

    def _run_bracket(
        self,
        iteration: int,
        bracket: int,
        n: int,
        r: float,
        scheduler: HyperbandScheduler,
    ) -> None:
        if self.verbose:
            print(f"\nBracket s={bracket}: n={n}, r={r:.4g}")

        configs, _scores = self._build_candidate_pool(bracket=bracket, n=n, budget=r)
        rungs = scheduler.successive_halving(bracket, n, r)
        self._run_asha_bracket(
            iteration=iteration,
            bracket=bracket,
            initial_configs=configs,
            rungs=rungs,
        )

    def _run_asha_bracket(
        self,
        iteration: int,
        bracket: int,
        initial_configs: list[dict[str, Any]],
        rungs: list[tuple[int, int, float]],
    ) -> None:
        """
        Async Successive Halving (ASHA): promote a config from rung i to i+1
        as soon as its in-rung rank is in the top 1/eta. Replaces the
        synchronous wait-all-then-halve loop.
        """
        if not rungs or not initial_configs:
            return

        # rung_results[i] -> list of (config, predicted_loss) at budget rungs[i].r
        rung_results: list[list[tuple[dict[str, Any], float]]] = [[] for _ in rungs]
        # Track which configs have already been promoted out of rung i.
        rung_promoted_keys: list[set[str]] = [set() for _ in rungs]
        rung_budgets = [r_i for _, _, r_i in rungs]
        rung_targets = [n_i for _, n_i, _ in rungs]
        max_rung = len(rungs) - 1

        # Pending evaluations to submit: (rung_idx, config).
        pending: list[tuple[int, dict[str, Any]]] = [
            (0, cfg) for cfg in initial_configs[: rung_targets[0]]
        ]

        in_flight: dict[concurrent.futures.Future, tuple[int, dict[str, Any]]] = {}
        executor: concurrent.futures.ThreadPoolExecutor | None = None
        max_workers = max(1, self.parallel_jobs)
        if max_workers > 1:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        def _record_completion(
            rung_idx: int, config: dict[str, Any], loss: float
        ) -> None:
            self.tpe.observe(config, loss, budget=rung_budgets[rung_idx])
            self._update_best(config, loss)
            predicted = self._predict_final_loss(loss, rung_budgets[rung_idx])
            rung_results[rung_idx].append((config, predicted))
            self._update_top_configs([(config, loss)])
            if rung_idx < max_rung:
                finished = rung_results[rung_idx]
                n_keep = max(1, len(finished) // self.eta)
                ranked = sorted(finished, key=lambda x: x[1])
                for cand_cfg, _ in ranked[:n_keep]:
                    cand_key = _canonical_config_key(cand_cfg)
                    if cand_key in rung_promoted_keys[rung_idx]:
                        continue
                    rung_promoted_keys[rung_idx].add(cand_key)
                    pending.append((rung_idx + 1, cand_cfg))

        try:
            while pending or in_flight:
                # Fill the in-flight pool up to max_workers, skipping configs
                # that fail hard constraints or hit the cache.
                while pending and len(in_flight) < max_workers:
                    rung_idx, config = pending.pop(0)
                    budget = rung_budgets[rung_idx]

                    if not self.tpe._hard_constraints_satisfied(config):
                        if self.verbose:
                            print("    Hard constraint violated; skipping.")
                        continue

                    cache_key = self._cache_key(config, budget)
                    if cache_key in self.config_cache:
                        if self.verbose:
                            print("    (cache hit)")
                        _record_completion(
                            rung_idx, config, self.config_cache[cache_key]
                        )
                        continue

                    if executor is not None:
                        fut = executor.submit(self._evaluate_objective, config, budget)
                    else:
                        fut = _ImmediateFuture(self._evaluate_objective(config, budget))
                    in_flight[fut] = (rung_idx, config)

                if not in_flight:
                    break

                if executor is not None:
                    done_iter = concurrent.futures.as_completed(list(in_flight))
                    done_fut = next(done_iter)
                else:
                    done_fut = next(iter(in_flight))

                rung_idx, config = in_flight.pop(done_fut)
                budget = rung_budgets[rung_idx]
                cache_key = self._cache_key(config, budget)

                try:
                    outcome = done_fut.result()
                except Exception as exc:
                    if self.handle_errors:
                        if self.verbose:
                            print(f"    Eval error in parallel job: {exc}")
                        outcome = None
                    else:
                        raise

                loss = self._finalize_evaluation(
                    config=config,
                    budget=budget,
                    outcome=outcome,
                    cache_key=cache_key,
                    iteration=iteration,
                    bracket=bracket,
                    round_idx=rung_idx,
                )
                if loss is None:
                    continue

                _record_completion(rung_idx, config, loss)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    def _build_candidate_pool(
        self, bracket: int, n: int, budget: float
    ) -> tuple[list[dict[str, Any]], list[float]]:
        recent_bracket_success = self._recent_bracket_improvement(bracket)
        pool_multiplier = 4 + 8 * (1.0 - recent_bracket_success)
        pool_n = max(n * int(pool_multiplier), 32, n * 3)

        configs, scores = self.tpe.suggest(
            n_candidates=pool_n,
            budget=budget,
            return_scores="aligned",
        )

        if budget >= 0.4 * self.max_budget and self.best_config:
            n_jitter = 1 if budget >= 0.7 * self.max_budget else 2
            for _ in range(n_jitter):
                jitter_cfg = self._jitter_config(
                    self.best_config,
                    scale=0.08 * (1.0 - budget / self.max_budget),
                )
                if self.tpe._hard_constraints_satisfied(jitter_cfg):
                    configs.append(jitter_cfg)
            configs = list({_canonical_config_key(c): c for c in configs}.values())

        if len(configs) > n:
            scored = [
                (cfg, float(scores[i]) if i < len(scores) else 0.0)
                for i, cfg in enumerate(configs)
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            configs = [cfg for cfg, _ in scored[:n]]
        else:
            configs = configs[:n]

        return configs, []

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

    def _build_rung_grid(self) -> list[float]:
        if self.min_budget <= 0 or self.max_budget <= 0 or self.eta < 2:
            return []
        s_max = int(math.log(self.max_budget / self.min_budget, self.eta))
        return [float(self.max_budget * (self.eta ** (-i))) for i in range(s_max + 1)]

    def _snap_budget(self, budget: float) -> float:
        """Round budget to ~9 sig figs, then snap to the nearest Hyperband
        rung if within 0.1% relative tolerance. Out-of-grid budgets (e.g.
        from arbitrary prior-trial JSONL) pass through with rounding only."""
        b = float(budget)
        if b <= 0 or not self._rung_budgets:
            return float(f"{b:.9g}")
        nearest = min(self._rung_budgets, key=lambda r: abs(r - b))
        if abs(nearest - b) <= 1e-3 * max(nearest, b):
            return nearest
        return float(f"{b:.9g}")

    def _cache_key(self, config: dict[str, Any], budget: float) -> tuple[str, float]:
        key_json = _canonical_config_key(config)
        key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()
        return key_hash, self._snap_budget(budget)

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

        if (
            not isinstance(raw_loss, (int, float))
            or np.isnan(raw_loss)
            or np.isinf(raw_loss)
        ):
            if self.handle_errors:
                if self.verbose:
                    print(f"    Invalid loss returned: {raw_loss}")
                return None
            raise ValueError(f"Invalid loss: {raw_loss}")

        loss = float(raw_loss)
        penalty = 0.0
        if self.tpe.soft_constraints:
            penalty = self.tpe._soft_constraint_violation(config)
            if penalty > 0:
                loss = loss + float(self.tpe.soft_penalty_weight) * penalty
        return loss, float(raw_loss), float(penalty)

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
        n = len(self.history)
        if (
            self._slope_history_len >= 0
            and n - self._slope_history_len < self._slope_refresh_every
        ):
            return self._slope_cached

        slope = self._fit_learning_curve_slope()
        self._slope_cached = slope
        self._slope_history_len = n
        return slope

    def _fit_learning_curve_slope(self) -> float:
        """
        Fit learning curve slope using Bayesian weighted regression.

        Upgraded from simple OLS to:
        1. Huber loss robust regression (resistant to outliers)
        2. Time-weighted observations (recent data weighted more)
        3. Budget-stratified sampling (ensure coverage across budgets)
        4. Bayesian posterior mean of slope (with uncertainty)

        Returns:
            Estimated learning curve slope in log-log space.
            Negative = loss decreases with budget (expected).
            0.0 = no clear learning signal.
        """
        if self.max_budget <= 0:
            return 0.0

        obs = [
            (h["budget"], h["loss"])
            for h in self.history
            if h.get("budget", 0) > 0 and h.get("loss", 0) > 0
        ]
        if len(obs) < 15:  # Lowered threshold since we weight better
            return 0.0

        budgets = np.array([b for b, _ in obs], dtype=float)
        losses = np.array([l for _, l in obs], dtype=float)

        # Filter to positive, finite values
        valid = (
            np.isfinite(budgets) & np.isfinite(losses) & (budgets > 0) & (losses > 0)
        )
        budgets = budgets[valid]
        losses = losses[valid]

        if len(budgets) < 15:
            return 0.0

        n_unique = len(np.unique(budgets))
        if n_unique < 2:
            # If only one budget level, estimate from quantile of losses
            losses_sorted = np.sort(losses)
            n_good = max(1, len(losses_sorted) // 10)
            if n_good < 2:
                return 0.0
            return 0.0  # Can't estimate slope from single budget

        x = np.log(budgets)
        y = np.log(losses)

        try:
            # --- Step 1: Outlier rejection via iterative re-weighting ---
            # Start with OLS, then iteratively re-weight based on residual
            k_prev, b_prev = np.polyfit(x, y, deg=1)
            residual = np.abs(y - (k_prev * x + b_prev))
            mad = np.median(residual)
            # Huber weights: down-weight large residuals
            weights = np.ones(len(x))
            if mad > 1e-8:
                z = residual / (1.4826 * mad)  # Normalized MAD
                weights = np.where(
                    z <= 1.5,
                    1.0,
                    1.5 / z,  # Huber weight for outliers
                )

            # --- Step 2: Time decay weighting ---
            # Recent observations should influence slope more
            time_weights = np.exp(-0.02 * np.arange(len(x)))
            weights = weights * time_weights

            # --- Step 3: Weighted robust regression ---
            k, b = np.polyfit(x, y, deg=1, w=weights)

            # --- Step 4: Bayesian-style posterior correction ---
            # Estimate slope uncertainty and shrink toward zero if uncertain
            predictions = k * x + b
            resid_sq = weights * (y - predictions) ** 2
            se = np.sqrt(np.sum(resid_sq) / max(len(x) - 2, 1))
            x_mean = np.mean(x)
            x_var = np.sum(weights * (x - x_mean) ** 2) / np.sum(weights)
            se_k = se / np.sqrt(max(x_var * len(x), 1e-12))

            # Bayesian shrinkage: pull estimate toward zero proportional to uncertainty
            prior_precision = 1.0  # Weak prior (slope ~ N(0, 1))
            data_precision = 1.0 / max(se_k**2, 1e-8)
            posterior_precision = prior_precision + data_precision
            k_shrunk = (
                prior_precision * 0.0 + data_precision * k
            ) / posterior_precision

            # --- Step 5: Conservative bounds ---
            k_final = float(k_shrunk)
            k_final = float(max(min(k_final, 0.0), -0.5))  # Allow steeper slopes
            if abs(k_final) < 0.015:  # Slightly lowered threshold
                return 0.0
            return k_final

        except Exception:
            # Fallback: simple OLS with outlier rejection
            try:
                k, _ = np.polyfit(x, y, deg=1)
                if np.isfinite(k):
                    return float(max(min(k, 0.0), -0.3))
            except Exception:
                pass
            return 0.0

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
        progress = self._step_progress(step, trial.budget)
        record = {
            "step": float(step),
            "budget": float(trial.budget),
            "progress": progress,
            "loss": float(loss),
        }
        step_key = int(step)
        progress_key = int(progress / self._progress_bin_width)
        entry = (float(trial.budget), float(progress), float(loss))
        with self._step_history_lock:
            self._step_history.append(record)
            self._step_by_step.setdefault(step_key, []).append(entry)
            self._step_by_progress_bin.setdefault(progress_key, []).append(entry)

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

        bin_radius = max(
            1, int(math.ceil(cfg.step_progress_tolerance / self._progress_bin_width))
        )
        target_bin = int(target_progress / self._progress_bin_width)

        with self._step_history_lock:
            step_bucket = list(self._step_by_step.get(int(step), ()))
            nearby_bucket: list[tuple[float, float, float]] = []
            for offset in range(-bin_radius, bin_radius + 1):
                nearby_bucket.extend(
                    self._step_by_progress_bin.get(target_bin + offset, ())
                )

        exact_matches = [
            loss for b, _p, loss in step_bucket if _budget_is_compatible(b)
        ]
        if len(exact_matches) >= cfg.step_exact_match_min:
            return exact_matches

        nearby_matches = [
            loss
            for b, p, loss in nearby_bucket
            if abs(p - target_progress) <= cfg.step_progress_tolerance
            and _budget_is_compatible(b)
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
