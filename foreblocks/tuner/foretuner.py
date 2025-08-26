import warnings

# ============================================
# ✅ Core Python & Concurrency
# ============================================
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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

class Foretuner:
    """
    Enhanced TURBO-M++ optimizer with unified acquisition and candidate management.

    Notes on behavior preservation:
    - All feature flags keep their same default values and meaning.
    - Loop cadence, region management, candidate generation, evaluation,
      progress tracking, and stopping criteria are unchanged.
    - Now uses unified AcquisitionCandidateManager for better performance
      and reduced code duplication.
    """

    def __init__(self, config: "TurboConfig" = None) -> None:
        self.config = config or TurboConfig()
        self.surrogate_manager = SurrogateManager(self.config)
        self.region_manager = RegionManager(self.config)
        
        # NEW: Unified acquisition and candidate manager
        self.acquisition_candidate_manager = AcquisitionCandidateManager(self.config)

        # Connect managers
        self.region_manager.set_surrogate_manager(self.surrogate_manager)

        # Core state
        self.global_X: Optional[np.ndarray] = None
        self.global_y: Optional[np.ndarray] = None
        self.iteration: int = 0
        self.global_best_history: List[float] = []
        self.stagnation_counter: int = 0
        self.last_global_improvement: int = 0

        # ---- Feature flags / knobs (defaults preserve original behavior) ----
        self._dup_tol: float = float(self._cfg("duplicate_tol", 0.0))                 # 0 = OFF
        self._penalty: float = float(self._cfg("failure_penalty", 1e12))              # used only if objective fails
        self._early_patience: int = int(self._cfg("early_stop_patience", 10))         # 0 = OFF
        self._sur_cadence: int = int(self._cfg("surrogate_update_cadence", 1))        # 1 = update every loop
        self._adaptive_mgmt: bool = bool(self._cfg("adaptive_management", False))     # OFF
        self._sanitize_before_eval: bool = bool(self._cfg("sanitize_candidates", True))  # ON (safe)
        self._dedupe_new: bool = bool(self._cfg("dedupe_new_candidates", True))       # ON

        # Cached base management frequency (can be adapted per-iteration)
        self._mgmt_freq_base: int = int(self._cfg("management_frequency", 5))

    # ===========================
    # Public API
    # ===========================
    @property
    def regions(self) -> Sequence:
        """Access regions through the manager."""
        return self.region_manager.regions

    # Legacy properties for backward compatibility
    @property
    def acquisition_manager(self):
        """Legacy access to acquisition functionality."""
        return self.acquisition_candidate_manager
    
    @property
    def candidate_generator(self):
        """Legacy access to candidate generation functionality."""
        return self.acquisition_candidate_manager

    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        seed: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Run the main optimization loop.

        Parameters
        ----------
        objective_fn : Callable[[np.ndarray], float]
            Black-box objective to minimize. Must return a scalar float.
        bounds : np.ndarray
            Shape (n_dims, 2) array with [low, high] per dimension.
        seed : int
            RNG seed for reproducibility of initialization and hygiene.

        Returns
        -------
        (x_best, y_best) : (np.ndarray, float)
            Best solution found and its objective value.
        """
        bounds = np.asarray(bounds, dtype=float)
        assert bounds.ndim == 2 and bounds.shape[1] == 2, "bounds must be (n_dims, 2)"
        n_dims: int = bounds.shape[0]

        rng = np.random.default_rng(seed)

        # === Initialization ===
        self._initialize_optimization(objective_fn, bounds, rng, n_dims)

        # === Main optimization loop ===
        # range(start, stop, step) matches original stride (batch_size)
        for self.iteration in range(self.config.n_init, self.config.max_evals, self.config.batch_size):
            self._update_context()

            # Region management cadence (fixed unless adaptive_management=True)
            mgmt_freq = self._mgmt_freq_base
            if self._adaptive_mgmt and self.stagnation_counter > 10:
                mgmt_freq = max(1, mgmt_freq // 2)
            if mgmt_freq > 0 and (self.iteration % mgmt_freq == 0):
                self.region_manager.manage_regions(
                    bounds, n_dims, rng, self.global_X, self.global_y, iteration=self.iteration
                )

            # Candidate generation using unified manager
            active_regions = [r for r in self.regions if getattr(r, "is_active", True)]
            candidates = self.acquisition_candidate_manager.generate_candidates(
                bounds, rng, active_regions, self.surrogate_manager
            )

            # Fallback to uniform random if generator yields nothing (should rarely happen now)
            if candidates is None or len(candidates) == 0:
                print(f"[Warning] Empty candidates at iteration {self.iteration}, falling back to random")
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

            # Notify unified manager (includes both acquisition and candidate components)
            try:
                self.acquisition_candidate_manager.notify_iteration_result(
                    improvement, 
                    region_id=id(active_regions[0]) if active_regions else None,
                    acquisition_used=getattr(self.acquisition_candidate_manager, '_last_strategy', None)
                )
            except Exception as e:
                # More informative error logging
                if hasattr(self.config, 'verbose') and self.config.verbose:
                    print(f"[Warning] Failed to notify unified manager: {e}")

            # Progress log
            if (self.iteration % 10) == 0:
                self._print_progress()

            # Early stop (opt-in)
            if self._early_patience > 0 and self.stagnation_counter >= self._early_patience:
                print(f"[EarlyStop] No improvement for {self.stagnation_counter} steps. Stopping.")
                break

        return self._get_best_solution()

    def get_trials(self) -> List["Trial"]:
        """Return all trials as Trial objects (for plotting/compat)."""
        trials: List[Trial] = []
        if self.global_X is None or self.global_y is None:
            return trials
        X, y = self.global_X, self.global_y
        for i in range(len(X)):
            params = {f"x{j}": X[i, j] for j in range(X.shape[1])}
            trials.append(Trial(params=params, value=float(y[i]), is_feasible=True))
        return trials

    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics from all managers."""
        diagnostics = {
            "iteration": self.iteration,
            "stagnation_counter": self.stagnation_counter,
            "global_best": float(np.min(self.global_y)) if self.global_y is not None else None,
            "n_evaluations": len(self.global_y) if self.global_y is not None else 0,
            "n_regions": len(self.regions) if self.regions else 0,
            "active_regions": sum(1 for r in self.regions if getattr(r, "is_active", True)) if self.regions else 0,
        }
        
        # Add unified manager diagnostics
        try:
            diagnostics["unified_manager"] = self.acquisition_candidate_manager.get_info()
        except Exception:
            diagnostics["unified_manager"] = {"error": "Failed to get diagnostics"}
        
        # Add region diagnostics
        if self.regions:
            region_healths = [getattr(r, "health_score", 0.0) for r in self.regions]
            region_radii = [getattr(r, "radius", 0.0) for r in self.regions]
            diagnostics["region_stats"] = {
                "mean_health": float(np.mean(region_healths)),
                "mean_radius": float(np.mean(region_radii)),
                "health_std": float(np.std(region_healths)),
                "radius_std": float(np.std(region_radii)),
            }
        
        return diagnostics

    # ===========================
    # Initialization
    # ===========================
    def _initialize_optimization(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        rng: np.random.Generator,
        n_dims: int,
    ) -> None:
        """Initialize optimization state (samples, evals, managers)."""
        X_init = self._initialize_points(n_dims, bounds, rng)
        y_init = self._evaluate_batch(objective_fn, X_init, bounds)

        self.global_X = X_init
        self.global_y = y_init

        # Update managers with initial data
        self.surrogate_manager.update_data(self.global_X, self.global_y)
        self.region_manager.initialize_regions(X_init, y_init, n_dims, rng)

        best_y = float(np.min(self.global_y))
        self.global_best_history = [best_y]
        print(f"Initial best: {best_y:.6f}")
        
    def _initialize_points(
        self,
        n_dims: int,
        bounds: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        RESTORED to match original behavior:
        - Sobol of size n_init
        - plus random extra of size n_init//4
        - unique in unit cube
        - DO NOT trim back to n_init (may return > n_init)
        """
        n_init = int(self.config.n_init)
        low, high = bounds[:, 0], bounds[:, 1]

        if n_init <= 0:
            return rng.uniform(low, high, size=(1, n_dims))

        # Exactly like your original
        sobol_seed = int(rng.integers(0, 10000))
        sobol_part = sobol_sequence(n_init, n_dims, sobol_seed)  # size = n_init
        rand_extra = rng.uniform(0.0, 1.0, (n_init // 4, n_dims))  # size = n_init//4
        all_unit = np.vstack([sobol_part, rand_extra])

        # Unique (lexicographic) like before; importantly, do NOT trim to n_init
        all_unit = np.unique(all_unit, axis=0)

        # Rescale to bounds; may return > n_init rows (same as before)
        return low + all_unit * (high - low)

    # ===========================
    # Iteration Context
    # ===========================
    def _update_context(self) -> None:
        """Update context for unified manager (iteration + stagnation)."""
        self.acquisition_candidate_manager.set_context(self.iteration, self.stagnation_counter)

    # ===========================
    # Global Data Updates
    # ===========================
    def _update_global_data(self, candidates: np.ndarray, y_new: np.ndarray) -> None:
        """Append new data; update surrogate (by cadence) and regions."""
        if self.global_X is None or self.global_y is None:
            # Should not happen after initialization; keep a defensive fallback.
            self.global_X = np.asarray(candidates, dtype=float)
            self.global_y = np.asarray(y_new, dtype=float)
        else:
            self.global_X = np.vstack([self.global_X, candidates])
            self.global_y = np.append(self.global_y, y_new)

        # Retrain surrogate model (cadence; default = every loop)
        # Use integer "outer loop" count based on batch_size to keep parity with original
        if (self.iteration // max(1, self.config.batch_size)) % max(1, self._sur_cadence) == 0:
            self.surrogate_manager.update_data(self.global_X, self.global_y)

        # Update trust regions
        self.region_manager.update_regions_with_new_data(candidates, y_new)

    # ===========================
    # Progress Tracking
    # ===========================
    def _track_progress(self) -> float:
        """
        Track optimization progress & stagnation counter; returns improvement magnitude.
        Behavior preserved exactly w.r.t. thresholds and counters.
        """
        current_best = float(np.min(self.global_y))
        prev_best = self.global_best_history[-1] if self.global_best_history else np.inf
        improvement = max(0.0, prev_best - current_best)

        if improvement > 1e-6:
            self.last_global_improvement = 0
            self.stagnation_counter = 0
        else:
            self.last_global_improvement += 1
            if self.last_global_improvement > 3:
                self.stagnation_counter = min(self.stagnation_counter + 1, 1000)

        self.global_best_history.append(current_best)
        return improvement

    # ===========================
    # Logging
    # ===========================
    def _print_progress(self) -> None:
        """Print optimization progress with unified manager info."""
        best_y = float(np.min(self.global_y))
        regs = list(self.regions) if self.regions is not None else []
        active_regions = sum(1 for r in regs if getattr(r, "is_active", True))
        
        # Region stats
        if regs:
            health_vals = [float(getattr(r, "health_score", 0.0)) for r in regs]
            radius_vals = [float(getattr(r, "radius", 0.0)) for r in regs]
            avg_health = float(np.mean(health_vals))
            avg_radius = float(np.mean(radius_vals))
        else:
            avg_health = 0.0
            avg_radius = 0.0

        # Unified manager stats
        try:
            manager_info = self.acquisition_candidate_manager.get_info()
            cross_region_size = manager_info.get("cross_region_memory_size", 0)
            fast_ts_fitted = manager_info.get("fast_ts_fitted", False)
            ts_status = "RF" if fast_ts_fitted else "GP"
        except:
            cross_region_size = 0
            ts_status = "?"

        print(
            f"Trial {self.iteration:4d}: Best = {best_y:.6f}, "
            f"Active regions = {active_regions}, Avg health = {avg_health:.3f}, "
            f"Avg radius = {avg_radius:.4f}, Stagnation = {self.stagnation_counter}, "
            f"TS = {ts_status}, Cross-mem = {cross_region_size}"
        )

    # ===========================
    # Best Solution
    # ===========================
    def _get_best_solution(self) -> Tuple[np.ndarray, float]:
        """Return best solution found so far (x_best, y_best)."""
        assert self.global_X is not None and self.global_y is not None, "No data available."
        best_idx = int(np.argmin(self.global_y))
        return self.global_X[best_idx], float(self.global_y[best_idx])

    # ===========================
    # Candidate hygiene (unchanged)
    # ===========================
    def _sanitize_candidates(self, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """
        Clamp to bounds & replace NaN/Inf (safe; shouldn't change behavior if
        objective is well-behaved). Keeps same semantics as original.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != bounds.shape[0]:
            # Mild guard; fall back to re-sampling if shape is inconsistent
            rng = np.random.default_rng()
            low, high = bounds[:, 0], bounds[:, 1]
            return rng.uniform(low, high, size=(self.config.batch_size, bounds.shape[0]))

        low, high = bounds[:, 0], bounds[:, 1]
        np.clip(X, low, high, out=X)
        bad = ~np.all(np.isfinite(X), axis=1)
        if np.any(bad):
            n_bad = int(np.sum(bad))
            rng = np.random.default_rng()
            X[bad] = rng.uniform(low, high, size=(n_bad, bounds.shape[0]))
        return X

    def _filter_duplicates(
        self,
        X_new: np.ndarray,
        X_all: Optional[np.ndarray],
        bounds: np.ndarray,
        rng: np.random.Generator,
        tol: Optional[float] = None,
    ) -> np.ndarray:
        """
        Opt-in: remove (near-)duplicates in X_new w.r.t. X_all and replace with fresh samples.
        Uses KDTree when available and HAS_KDTREE is True; falls back to pairwise distances.
        """
        tol = self._dup_tol if (tol is None) else tol
        if tol <= 0 or X_all is None or len(X_all) == 0 or len(X_new) == 0:
            return X_new

        n_dims = bounds.shape[0]
        low, high = bounds[:, 0], bounds[:, 1]
        X_new = np.asarray(X_new, dtype=float)

        dup_mask = None
        if HAS_KDTREE and KDTree is not None:
            tree = KDTree(np.asarray(X_all, dtype=float))
            d, _ = tree.query(X_new, k=1)
            dup_mask = (d <= tol)
        else:
            from scipy.spatial.distance import cdist
            d = cdist(X_new, X_all)
            dup_mask = (np.min(d, axis=1) <= tol)

        if not np.any(dup_mask):
            return X_new

        n_dup = int(np.sum(dup_mask))
        repl = rng.uniform(low, high, size=(n_dup, n_dims))

        # single-pass recheck (KDTree path only)
        if HAS_KDTREE and KDTree is not None:
            d2, _ = tree.query(repl, k=1)
            hits = (d2 <= tol)
            if np.any(hits):
                repl[hits] = rng.uniform(low, high, size=(int(np.sum(hits)), n_dims))

        X_out = X_new.copy()
        X_out[dup_mask] = repl
        return X_out

    # ===========================
    # Safe evaluation (unchanged)
    # ===========================
    def _evaluate_batch(
        self,
        objective_fn: Callable[[np.ndarray], float],
        X: np.ndarray,
        bounds: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate objective with guardrails:
        - optional sanitize (clamp + NaN/Inf fix)
        - treat non-finite returns/Exceptions as failure with penalty
        """
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

    # ===========================
    # Helpers
    # ===========================
    def _cfg(self, name: str, default):
        """Typed, safe access to self.config attributes with a default."""
        return getattr(self.config, name, default)
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
