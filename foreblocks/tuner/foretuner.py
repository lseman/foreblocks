import warnings

# ============================================
# ✅ Core Python & Concurrency
# ============================================
from typing import Callable, Dict, List, Tuple

# ============================================
# ✅ Visualization
# ============================================
import matplotlib.pyplot as plt

# ============================================
# ✅ Numerical & Scientific Computing
# ============================================
import numpy as np
import pandas as pd

from .foretuner_acq import *

# ============================================
# ✅ Project-Specific
# ============================================
from .foretuner_aux import *
from .foretuner_candidate import *
from .foretuner_sur import *
from .foretuner_tr import *

warnings.filterwarnings("ignore")

try:
    import seaborn as sns
    from pandas.plotting import parallel_coordinates

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import mplcursors

    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False


class Trial:
    """Simple trial object to hold optimization data"""

    def __init__(
        self, params: Dict[str, float], value: float, is_feasible: bool = True
    ):
        self.params = params
        self.value = value
        self.is_feasible = is_feasible
        self.constraint_violations = []  # For compatibility

try:
    from pykdtree.kdtree import KDTree

    HAS_KDTREE = True
except Exception:
    HAS_KDTREE = False

# Assumes these exist in your module:
# - TurboConfig, SurrogateManager, RegionManager, AcquisitionManager, CandidateGenerator, Trial
# - sobol_sequence(seed, n, d)  -> your existing initializer helper


class Foretuner:
    """Enhanced TURBO-M++ optimizer with opt-in safety/speed features (defaults match original behavior)."""

    def __init__(self, config: TurboConfig = None):
        self.config = config or TurboConfig()
        self.surrogate_manager = SurrogateManager(self.config)
        self.region_manager = RegionManager(self.config)
        self.acquisition_manager = AcquisitionManager(self.config)
        self.candidate_generator = CandidateGenerator(self.config, self.acquisition_manager)

        # Connect managers
        self.region_manager.set_surrogate_manager(self.surrogate_manager)

        # Core state
        self.global_X = None
        self.global_y = None
        self.iteration = 0
        self.global_best_history = []
        self.stagnation_counter = 0
        self.last_global_improvement = 0

        # ---- Feature flags / knobs (defaults preserve original behavior) ----
        self._dup_tol = float(getattr(self.config, "duplicate_tol", 0.0))                 # 0 = OFF
        self._penalty = float(getattr(self.config, "failure_penalty", 1e12))              # used only if objective fails
        self._early_patience = int(getattr(self.config, "early_stop_patience", 10))        # 0 = OFF
        self._sur_cadence = int(getattr(self.config, "surrogate_update_cadence", 1))      # 1 = update every loop
        self._adaptive_mgmt = bool(getattr(self.config, "adaptive_management", False))    # OFF
        self._sanitize_before_eval = bool(getattr(self.config, "sanitize_candidates", True))      # ON (safe)
        self._dedupe_new = bool(getattr(self.config, "dedupe_new_candidates", True))     # OFF

    @property
    def regions(self):
        """Access regions through manager"""
        return self.region_manager.regions

    def optimize(self, objective_fn: Callable, bounds: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, float]:
        """Main optimization loop with cleaner separation of concerns"""
        n_dims = bounds.shape[0]
        rng = np.random.default_rng(seed)

        # === Initialization ===
        self._initialize_optimization(objective_fn, bounds, rng, n_dims)

        # === Main optimization loop ===
        for self.iteration in range(self.config.n_init, self.config.max_evals, self.config.batch_size):
            self._update_context()

            # Region management cadence (fixed unless adaptive_management=True)
            mgmt_freq = int(getattr(self.config, "management_frequency", 5))
            if self._adaptive_mgmt and self.stagnation_counter > 10:
                mgmt_freq = max(1, mgmt_freq // 2)
            if self.iteration % mgmt_freq == 0:
                self.region_manager.manage_regions(
                    bounds, n_dims, rng, self.global_X, self.global_y, iteration=self.iteration
                )

            # Candidate generation
            active_regions = [r for r in self.regions if r.is_active]
            candidates = self.candidate_generator.generate_candidates(
                bounds, rng, active_regions, self.surrogate_manager
            )
            if candidates is None or len(candidates) == 0:
                candidates = rng.uniform(bounds[:, 0], bounds[:, 1], size=(self.config.batch_size, n_dims))

            # Hygiene (opt-in dedupe; always clamp/NaN-fix unless disabled)
            if self._sanitize_before_eval:
                candidates = self._sanitize_candidates(candidates, bounds)
            if self._dedupe_new and self._dup_tol > 0:
                candidates = self._filter_duplicates(candidates, self.global_X, bounds, rng)

            # Evaluate objective on generated candidates (guarded)
            y_new = self._evaluate_batch(objective_fn, candidates, bounds)

            # Update global + surrogate + regions
            self._update_global_data(candidates, y_new)

            # Track progress
            improvement = self._track_progress()

            # Notify acquisition (best-effort; ignore errors)
            try:
                self.acquisition_manager.notify_iteration_result(improvement)
            except Exception:
                pass

            # Progress log
            if self.iteration % 10 == 0:
                self._print_progress()

            # Early stop (opt-in)
            if self._early_patience > 0 and self.stagnation_counter >= self._early_patience:
                print(f"[EarlyStop] No improvement for {self.stagnation_counter} steps. Stopping.")
                break

        return self._get_best_solution()

    # === Initialization ===
    def _initialize_optimization(self, objective_fn, bounds, rng, n_dims):
        """Initialize optimization state"""
        X_init = self._initialize_points(n_dims, bounds, rng)
        y_init = self._evaluate_batch(objective_fn, X_init, bounds)

        self.global_X = X_init
        self.global_y = y_init

        # Update managers with initial data
        self.surrogate_manager.update_data(self.global_X, self.global_y)
        self.region_manager.initialize_regions(X_init, y_init, n_dims, rng)

        best_y = float(np.min(self.global_y))
        self.global_best_history.append(best_y)
        print(f"Initial best: {best_y:.6f}")

    def _initialize_points(self, n_dims, bounds, rng):
        """Original-style init: Sobol + random extra (your helper sobol_sequence)."""
        n_init = self.config.n_init

        # Sobol sequence for half (uses your existing helper)
        sobol_part = sobol_sequence(n_init, n_dims, rng.integers(0, 10000))

        # Random extra points for diversity
        rand_extra = rng.uniform(0, 1, (n_init // 4, n_dims))

        # Combine and drop potential duplicates within the init pool
        all_samples = np.vstack([sobol_part, rand_extra])
        all_samples = np.unique(all_samples, axis=0)

        low, high = bounds[:, 0], bounds[:, 1]
        return low + all_samples * (high - low)

    # === Iteration Context ===
    def _update_context(self):
        """Update context for all managers"""
        self.candidate_generator.set_context(self.iteration, self.stagnation_counter)

    # === Global Data Updates ===
    def _update_global_data(self, candidates, y_new):
        """Update global data and regions"""
        # Append new data
        self.global_X = np.vstack([self.global_X, candidates])
        self.global_y = np.append(self.global_y, y_new)

        # Retrain surrogate model (cadence; default = every loop)
        if (self.iteration // max(1, self.config.batch_size)) % max(1, self._sur_cadence) == 0:
            self.surrogate_manager.update_data(self.global_X, self.global_y)

        # Update trust regions
        self.region_manager.update_regions_with_new_data(candidates, y_new)

    # === Progress Tracking ===
    def _track_progress(self):
        """Track optimization progress & stagnation counter; returns improvement magnitude."""
        prev_best = self.global_best_history[-1] if self.global_best_history else np.inf
        current_best_y = float(np.min(self.global_y))
        improvement = max(0.0, prev_best - current_best_y)

        if improvement > 1e-6:
            self.last_global_improvement = 0
            self.stagnation_counter = 0
        else:
            self.last_global_improvement += 1
            if self.last_global_improvement > 3:
                self.stagnation_counter = min(self.stagnation_counter + 1, 1000)

        self.global_best_history.append(current_best_y)
        return improvement

    # === Logging ===
    def _print_progress(self):
        """Print optimization progress safely (handles empty regions)"""
        best_y = float(np.min(self.global_y))
        active_regions = sum(1 for r in self.regions if r.is_active)
        if self.regions:
            avg_health = float(np.mean([r.health_score for r in self.regions]))
            avg_radius = float(np.mean([r.radius for r in self.regions]))
        else:
            avg_health = 0.0
            avg_radius = 0.0

        print(
            f"Trial {self.iteration:4d}: Best = {best_y:.6f}, "
            f"Active regions = {active_regions}, Avg health = {avg_health:.3f}, "
            f"Avg radius = {avg_radius:.4f}, Stagnation = {self.stagnation_counter}"
        )

    # === Best Solution ===
    def _get_best_solution(self):
        """Return best solution found so far"""
        best_idx = int(np.argmin(self.global_y))
        return self.global_X[best_idx], float(self.global_y[best_idx])

    def get_trials(self) -> List[Trial]:
        """Get all trials as Trial objects for compatibility with plotting"""
        trials = []
        for i in range(len(self.global_X)):
            params = {f"x{j}": self.global_X[i, j] for j in range(self.global_X.shape[1])}
            trial = Trial(params=params, value=float(self.global_y[i]), is_feasible=True)
            trials.append(trial)
        return trials

    # === Candidate hygiene ===
    def _sanitize_candidates(self, X, bounds):
        """Clamp to bounds & replace NaN/Inf (safe; shouldn’t change behavior if objective is well-behaved)."""
        X = np.asarray(X, dtype=float)
        low, high = bounds[:, 0], bounds[:, 1]
        X = np.clip(X, low, high)
        bad = ~np.all(np.isfinite(X), axis=1)
        if np.any(bad):
            n_bad = int(np.sum(bad))
            rng = np.random.default_rng()
            X[bad] = rng.uniform(low, high, size=(n_bad, bounds.shape[0]))
        return X

    def _filter_duplicates(self, X_new, X_all, bounds, rng, tol=None):
        """Opt-in: remove (near-)duplicates in X_new w.r.t. X_all and replace with fresh samples."""
        if tol is None:
            tol = self._dup_tol
        if tol <= 0 or X_all is None or len(X_all) == 0 or len(X_new) == 0:
            return X_new

        n_dims = bounds.shape[0]
        low, high = bounds[:, 0], bounds[:, 1]
        X_new = np.asarray(X_new, dtype=float)

        if HAS_KDTREE:
            tree = KDTree(X_all)
            d, _ = tree.query(X_new, k=1)
            dup_mask = d <= tol
        else:
            from scipy.spatial.distance import cdist
            d = cdist(X_new, X_all)
            dup_mask = (np.min(d, axis=1) <= tol)

        if not np.any(dup_mask):
            return X_new

        n_dup = int(np.sum(dup_mask))
        repl = rng.uniform(low, high, size=(n_dup, n_dims))

        # single pass recheck (KDTree path only)
        if HAS_KDTREE:
            d2, _ = tree.query(repl, k=1)
            hits = d2 <= tol
            if np.any(hits):
                repl[hits] = rng.uniform(low, high, size=(int(np.sum(hits)), n_dims))

        X_new = X_new.copy()
        X_new[dup_mask] = repl
        return X_new

    # === Safe evaluation ===
    def _evaluate_batch(self, objective_fn, X, bounds):
        """Evaluate objective with guardrails: clamps; handles NaN/inf/Exceptions."""
        X = self._sanitize_candidates(X, bounds) if self._sanitize_before_eval else np.asarray(X, dtype=float)
        y = np.empty(len(X), dtype=float)
        for i, xi in enumerate(X):
            try:
                val = objective_fn(np.asarray(xi, dtype=float))
                if not np.isfinite(val):
                    raise ValueError("non-finite objective")
                y[i] = float(val)
            except Exception:
                y[i] = self._penalty
        return y


def plot_foretuner_results(
    optimizer,
    bounds: np.ndarray,
    param_names: List[str] = None,
    title: str = "Foretuner Optimization Results",
):
    """
    Plot optimization results for Foretuner class

    Args:
        optimizer: Foretuner instance after optimization
        bounds: Parameter bounds array (n_dims x 2)
        param_names: List of parameter names (optional)
        title: Plot title
    """

    # Extract data from optimizer
    X = optimizer.global_X
    y = optimizer.global_y

    if param_names is None:
        param_names = [f"x{i}" for i in range(X.shape[1])]

    # Convert to trial objects for compatibility with existing plot function
    trials = []
    for i in range(len(X)):
        params = {param_names[j]: X[i, j] for j in range(len(param_names))}
        trial = Trial(params=params, value=y[i], is_feasible=True)
        trials.append(trial)

    # Use the existing plot function
    plot_optimization_results(trials, title)


def plot_optimization_results(trials: List, title: str = "Enhanced Foretuner Results"):
    """State-of-the-art optimization visualization for Foretuner trials"""

    feasible_trials = [t for t in trials if t.is_feasible]
    all_values = [t.value for t in trials]
    feasible_values = [t.value for t in feasible_trials]

    if not feasible_values:
        feasible_values = all_values
        print("⚠️ No feasible trials found, showing all trials")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Convergence Plot ---
    ax = axes[0, 0]
    ax.plot(all_values, "o-", alpha=0.4, label="All", color="lightblue")

    feasible_indices = [i for i, t in enumerate(trials) if t.is_feasible]
    ax.plot(
        feasible_indices,
        feasible_values,
        "o-",
        alpha=0.8,
        label="Feasible",
        color="blue",
    )

    best_values = [min(feasible_values[: i + 1]) for i in range(len(feasible_values))]
    ax.plot(feasible_indices, best_values, "r-", linewidth=3, label="Best Feasible")

    # Optional: Initial BO cutoff
    init_cutoff = len([t for t in trials if getattr(t, "is_initial", False)])
    if init_cutoff > 0:
        ax.axvline(init_cutoff, color="gray", linestyle="--", label="Start BO")

    ax.set_title(f"{title} - Convergence")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate best point
    if feasible_values:
        best_idx = feasible_indices[np.argmin(feasible_values)]
        best_val = min(feasible_values)
        ax.annotate(
            f"Best: {best_val:.4f}",
            xy=(best_idx, best_val),
            xytext=(10, -20),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=9,
            color="green",
        )

    # --- Log Convergence ---
    ax = axes[0, 1]
    best_values_pos = np.maximum(best_values, 1e-10)
    ax.semilogy(best_values_pos, "r-", linewidth=2)
    ax.set_title("Log Convergence")
    ax.set_xlabel("Feasible Trial")
    ax.set_ylabel("Best Value (log)")
    ax.grid(True, alpha=0.3)

    # --- Value Distribution (Histogram or KDE) ---
    ax = axes[0, 2]
    if len(feasible_values) > 1:
        try:
            if SEABORN_AVAILABLE:
                sns.kdeplot(feasible_values, ax=ax, fill=True, color="skyblue")
            else:
                ax.hist(
                    feasible_values,
                    bins="auto",
                    edgecolor="black",
                    alpha=0.7,
                    color="skyblue",
                )
            ax.axvline(
                min(feasible_values),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Best: {min(feasible_values):.4f}",
            )
            ax.legend()
        except Exception:
            ax.hist(
                feasible_values,
                bins="auto",
                edgecolor="black",
                alpha=0.7,
                color="skyblue",
            )
        ax.set_title("Objective Value Distribution")
    else:
        ax.text(
            0.5,
            0.5,
            f"Single Value:\n{feasible_values[0]:.4f}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    ax.set_xlabel("Objective Value")
    ax.set_ylabel("Density/Frequency")
    ax.grid(True, alpha=0.3)

    # --- Constraint Violations ---
    ax = axes[1, 0]
    constraint_counts = [
        len(t.constraint_violations) if hasattr(t, "constraint_violations") else 0
        for t in trials
    ]
    if any(constraint_counts):
        ax.plot(constraint_counts, "o-", color="orange", alpha=0.7)
        ax.set_title("Constraint Violations")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Violations")
    else:
        ax.text(
            0.5,
            0.5,
            "No Constraints\nor All Feasible",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Constraint Status")
    ax.grid(True, alpha=0.3)

    # --- Improvement Rate ---
    ax = axes[1, 1]
    if len(best_values) > 10:
        window = min(10, len(best_values) // 4)
        improvements = [
            (best_values[i - window] - best_values[i])
            / (abs(best_values[i - window]) + 1e-10)
            for i in range(window, len(best_values))
        ]
        ax.plot(range(window, len(best_values)), improvements, "g-", linewidth=2)
        ax.set_title("Improvement Rate")
        ax.set_xlabel(f"Trial (window: {window})")
        ax.set_ylabel("Relative Improvement")
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient Data\nfor Rate Analysis",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
    ax.grid(True, alpha=0.3)

    # --- Parameter Space: 1D or 2D ---
    ax = axes[1, 2]
    param_keys = list(trials[0].params.keys())

    if len(param_keys) >= 2:
        # === 2D case ===
        x_all = [t.params[param_keys[0]] for t in trials]
        y_all = [t.params[param_keys[1]] for t in trials]
        vals_all = [t.value for t in trials]
        feas_flags = [t.is_feasible for t in trials]

        x_feas = [x for x, f in zip(x_all, feas_flags) if f]
        y_feas = [y for y, f in zip(y_all, feas_flags) if f]
        val_feas = [v for v, f in zip(vals_all, feas_flags) if f]

        x_infeas = [x for x, f in zip(x_all, feas_flags) if not f]
        y_infeas = [y for y, f in zip(y_all, feas_flags) if not f]

        scatter = ax.scatter(
            x_feas,
            y_feas,
            c=val_feas,
            cmap="viridis_r",
            edgecolors="black",
            s=60,
            alpha=0.8,
            label="Feasible",
        )
        ax.scatter(
            x_infeas, y_infeas, marker="x", color="red", s=50, label="Infeasible"
        )

        if val_feas:
            best_idx = np.argmin(val_feas)
            ax.scatter(
                x_feas[best_idx],
                y_feas[best_idx],
                marker="*",
                s=200,
                c="gold",
                edgecolors="black",
                linewidths=1.5,
                label="Best",
            )

        plt.colorbar(scatter, ax=ax, label="Objective Value")
        ax.set_xlabel(param_keys[0])
        ax.set_ylabel(param_keys[1])
        ax.set_title("2D Parameter Space (colored by value)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif len(param_keys) == 1:
        # === 1D case ===
        x = [t.params[param_keys[0]] for t in trials]
        y = [t.value for t in trials]
        feas_flags = [t.is_feasible for t in trials]

        x_feas = [xi for xi, f in zip(x, feas_flags) if f]
        y_feas = [yi for yi, f in zip(y, feas_flags) if f]
        x_infeas = [xi for xi, f in zip(x, feas_flags) if not f]
        y_infeas = [yi for yi, f in zip(y, feas_flags) if not f]

        ax.scatter(
            x_feas,
            y_feas,
            c="blue",
            label="Feasible",
            edgecolors="black",
            alpha=0.7,
            s=60,
        )
        ax.scatter(x_infeas, y_infeas, c="red", marker="x", label="Infeasible", s=50)

        if y_feas:
            best_idx = np.argmin(y_feas)
            ax.scatter(
                x_feas[best_idx],
                y_feas[best_idx],
                marker="*",
                s=200,
                c="gold",
                edgecolors="black",
                linewidths=1.5,
                label="Best",
            )

        ax.set_xlabel(param_keys[0])
        ax.set_ylabel("Objective Value")
        ax.set_title("1D Parameter Plot")
        ax.legend()
        ax.grid(True, alpha=0.3)

    else:
        ax.text(
            0.5,
            0.5,
            "No parameters to visualize",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Parameter Space")

    plt.tight_layout()
    plt.show()

    # --- Optional: Parallel Coordinates if >2D ---
    if len(trials[0].params) > 2 and SEABORN_AVAILABLE:
        try:
            df = pd.DataFrame(
                [dict(**t.params, value=t.value) for t in feasible_trials]
            )
            df["label"] = pd.qcut(df["value"], q=3, labels=["High", "Medium", "Low"])
            plt.figure(figsize=(12, 6))
            parallel_coordinates(
                df[["label"] + list(trials[0].params.keys())],
                class_column="label",
                colormap="coolwarm",
                alpha=0.6,
            )
            plt.title("Parallel Coordinates (Parameter Patterns)")
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e:
            print(f"Could not create parallel coordinates plot: {e}")

    # --- Optional: Parameter Correlation Heatmap ---
    if SEABORN_AVAILABLE:
        try:
            df_params = pd.DataFrame([t.params for t in feasible_trials])
            if not df_params.empty and len(df_params.columns) > 1:
                df_params["value"] = feasible_values
                plt.figure(figsize=(10, 6))
                sns.heatmap(df_params.corr(), annot=True, fmt=".2f", cmap="coolwarm")
                plt.title("Parameter Correlation Matrix")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Could not create correlation heatmap: {e}")

    # --- Optional: Interactive hover ---
    if MPLCURSORS_AVAILABLE:
        mplcursors.cursor(hover=True)

    # --- Summary Statistics ---
    print("\n📊 Optimization Summary:")
    print(f"   Total trials: {len(trials)}")
    print(
        f"   Feasible trials: {len(feasible_trials)} ({len(feasible_trials) / len(trials) * 100:.1f}%)"
    )
    print(f"   Best value: {min(feasible_values):.6f}")
    print(f"   Value range: {max(feasible_values) - min(feasible_values):.6f}")
    if len(best_values) > 10:
        final_improv = (best_values[-10] - best_values[-1]) / (
            abs(best_values[-10]) + 1e-10
        )
        print(f"   Final convergence rate (last 10): {final_improv:.4f}")
