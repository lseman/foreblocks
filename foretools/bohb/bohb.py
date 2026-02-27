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

import hashlib
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


from .utils import safe_log
from .tpe import TPE, TPEConf
from .utils import _canonical_config_key

# -----------------------------------------------------------------------------
# BOHB (Hyperband + TPE proposer)
# -----------------------------------------------------------------------------


class BOHB:
    def __init__(
        self,
        config_space: Dict[str, Tuple],
        evaluate_fn: Callable[[Dict[str, Any], float], float],
        min_budget: float = 1.0,
        max_budget: float = 81.0,
        eta: int = 3,
        n_iterations: int = 10,
        verbose: bool = True,
        handle_errors: bool = True,
        top_n_percent: int = 15,
        prior_trials_jsonl: Optional[str] = None,
        pruning_mode: str = "conservative",
        history_export_jsonl: Optional[str] = None,
        early_prune: bool = True,
        seed: Optional[int] = None,
        tpe_conf: Optional[TPEConf] = None,
        tpe_overrides: Optional[Dict[str, Any]] = None,
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

        self.history: List[Dict[str, Any]] = []
        self.best_loss = float("inf")
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_configs: List[Tuple[Dict[str, Any], float]] = []

        # cache: (sha1(config), budget) -> loss
        self.config_cache: Dict[Tuple[str, float], float] = {}

        if prior_trials_jsonl:
            self._load_prior_trials_jsonl(prior_trials_jsonl)

    def run(self) -> Tuple[Dict[str, Any], float]:
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

                    results: List[Tuple[Dict[str, Any], float]] = []
                    # Only evaluate as many as needed; configs list is already at least n.
                    for cfg in configs[:n_i]:
                        loss = self._evaluate_with_cache(cfg, r_i, it, s, i)
                        if loss is None:
                            continue

                        results.append((cfg, loss))
                        self.tpe.observe(cfg, loss, budget=r_i)

                        if loss < self.best_loss:
                            self.best_loss = float(loss)
                            self.best_config = dict(cfg)
                            if self.verbose:
                                print(
                                    f"    New best: {self.best_loss:.6g}  cfg={self.best_config}"
                                )

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

    def _evaluate_with_cache(
        self,
        config: Dict[str, Any],
        budget: float,
        iteration: int,
        bracket: int,
        round_idx: int,
    ) -> Optional[float]:
        if not self.tpe._hard_constraints_satisfied(config):
            if self.verbose:
                print("    Hard constraint violated; skipping.")
            return None

        key_json = _canonical_config_key(config)
        key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()
        cache_key = (key_hash, float(budget))

        if cache_key in self.config_cache:
            if self.verbose:
                print("    (cache hit)")
            return self.config_cache[cache_key]

        try:
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

            if self.early_prune and self._should_prune(loss, budget):
                if self.verbose:
                    print("    Early prune (loss above historical quantile).")
                return None

            self.config_cache[cache_key] = loss
            self.history.append(
                {
                    "config": dict(config),
                    "budget": float(budget),
                    "loss": loss,
                    "raw_loss": float(raw_loss),
                    "constraint_violation": float(penalty),
                    "iteration": int(iteration),
                    "bracket": int(bracket),
                    "round": int(round_idx),
                }
            )
            return loss
        except Exception as e:
            if self.handle_errors:
                if self.verbose:
                    print(f"    Eval error: {e}")
                return None
            raise

    def _update_top_configs(self, results: List[Tuple[Dict[str, Any], float]]) -> None:
        for cfg, loss in results:
            self.best_configs.append((dict(cfg), float(loss)))
        self.best_configs.sort(key=lambda x: x[1])
        n_keep = max(10, int(len(self.best_configs) * self.top_n_percent / 100))
        self.best_configs = self.best_configs[:n_keep]

    def _load_prior_trials_jsonl(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
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
                    key_json = _canonical_config_key(cfg)
                    key_hash = hashlib.sha1(key_json.encode("utf-8")).hexdigest()
                    self.config_cache[(key_hash, float(budget))] = loss
                    self.history.append(
                        {
                            "config": dict(cfg),
                            "budget": float(budget),
                            "loss": float(loss),
                            "raw_loss": float(loss),
                            "constraint_violation": 0.0,
                            "iteration": -1,
                            "bracket": -1,
                            "round": -1,
                        }
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
        if not self.early_prune or current_budget <= self.min_budget * 2:
            return False

        relevant = [
            h["loss"]
            for h in self.history
            if h.get("budget", 0) <= current_budget
            and h.get("loss", float("inf")) < float("inf")
        ]
        if len(relevant) < 8:
            return False

        progress = current_budget / self.max_budget
        q = 0.75 + 0.20 * progress
        q = min(max(q, 0.70), 0.92)

        threshold = float(np.quantile(relevant, q))
        if current_loss <= threshold:
            return False

        relative_worst = (current_loss - threshold) / (np.std(relevant) + 1e-6)
        if self.pruning_mode == "aggressive":
            p_base = 0.85
        elif self.pruning_mode == "balanced":
            p_base = 0.65
        else:
            p_base = 0.4
        p = min(0.95, p_base + 0.25 * max(0.0, relative_worst - 1.0))

        roll = self._rng.random()
        if self.verbose and roll < p:
            print(
                f"  → Early prune triggered (loss={current_loss:.6g} > quantile {q:.2f}={threshold:.6g}, p={p:.2f})"
            )
        return roll < p

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        return list(self.history)

    def get_top_configs(self, n: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        return list(self.best_configs[: int(n)])

    def _jitter_config(
        self, config: Dict[str, Any], scale: float = 0.08
    ) -> Dict[str, Any]:
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
                    new_val = val_log + np.random.normal(0, scale * (hi_log - lo_log))
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


# -----------------------------------------------------------------------------
# Example objective
# -----------------------------------------------------------------------------


def realistic_nn_objective(config: Dict[str, Any], budget: float) -> float:
    """
    Simulate training dynamics:
    - Some hyperparameter optimum
    - Diminishing returns with budget
    - Noise decreases with budget
    """
    lr = float(config["lr"])
    batch_size = int(config.get("batch_size", 32))
    dropout = float(config.get("dropout", 0.0))

    optimal_lr = 1e-2
    optimal_batch_size = 64
    optimal_dropout = 0.3

    lr_penalty = (math.log10(lr) - math.log10(optimal_lr)) ** 2
    bs_penalty = ((batch_size - optimal_batch_size) ** 2) / 1000.0
    do_penalty = (dropout - optimal_dropout) ** 2

    base_loss = lr_penalty + bs_penalty + do_penalty + 0.1

    improvement = 1.0 / (1.0 + 0.5 * safe_log(budget))
    noise = np.random.normal(0.0, 0.02 / math.sqrt(max(budget, 1e-12)))

    final_loss = base_loss * improvement + float(noise)
    return max(1e-3, float(final_loss))


def torch_mlp_objective(config: Dict[str, Any], budget: float) -> float:
    """
    Small Torch MLP objective for BOHB demo.
    budget controls epochs (rounded to int >= 1).
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install torch to use this objective."
        )

    torch.manual_seed(42)
    np.random.seed(42)

    lr = float(config["lr"])
    hidden = int(config.get("hidden", 64))
    dropout = float(config.get("dropout", 0.2))
    batch_size = int(config.get("batch_size", 64))

    # Synthetic regression data
    n_train = 1024
    n_val = 256
    n_features = 20
    X = torch.randn(n_train + n_val, n_features)
    true_w = torch.randn(n_features, 1) * 0.5
    y = X @ true_w + 0.1 * torch.randn(n_train + n_val, 1)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    model = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, 1),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epochs = max(1, int(round(float(budget))))
    model.train()
    for _ in range(epochs):
        idx = torch.randint(0, n_train, (batch_size,))
        xb = X_train[idx]
        yb = y_train[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
    return float(val_loss)


# -----------------------------------------------------------------------------
# Run example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    config_space = {
        "lr": ("float", (1e-5, 1e-1, "log")),
        "batch_size": ("choice", [16, 32, 64, 128, 256]),
        "dropout": ("float", (0.0, 0.5)),
        "hidden": ("int", (16, 256)),
    }

    bohb = BOHB(
        config_space=config_space,
        evaluate_fn=torch_mlp_objective,
        min_budget=3,
        max_budget=81,
        eta=3,
        n_iterations=3,
        verbose=True,
        handle_errors=True,
        top_n_percent=15,
        tpe_overrides={"n_startup_trials": 10, "n_ei_candidates": 64, "gamma": 0.15},
    )

    best_cfg, best_loss = bohb.run()

    from plotter import OptimizationPlotter

    plotter = OptimizationPlotter.from_bohb(bohb)
    plotter.plot_optimization_history(save_path="fig_history.png")
    plotter.plot_budget_vs_loss(save_path="fig_budget.png")
    plotter.plot_bracket_best(save_path="fig_bracket.png")
    plotter.plot_param_effect("lr", save_path="fig_param_lr.png")
    plotter.plot_parallel_coordinates(save_path="fig_parallel.png")
    plotter.plot_param_importance(save_path="fig_importance.png")

    print("\nTop 5 configurations found:")
    for i, (cfg, loss) in enumerate(bohb.get_top_configs(5), 1):
        print(f"{i}. loss={loss:.6f}  cfg={cfg}")
