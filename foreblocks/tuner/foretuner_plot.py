from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import parallel_coordinates

# Optional: mplcursors for interactive plots (Jupyter)
try:
    import mplcursors

    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

from .foretuner import *


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
            sns.kdeplot(feasible_values, ax=ax, fill=True, color="skyblue")
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
    if len(trials[0].params) > 2:
        df = pd.DataFrame([dict(**t.params, value=t.value) for t in feasible_trials])
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

    # --- Optional: Parameter Correlation Heatmap ---
    df_params = pd.DataFrame([t.params for t in feasible_trials])
    if not df_params.empty:
        df_params["value"] = feasible_values
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_params.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Parameter Correlation Matrix")
        plt.tight_layout()
        plt.show()

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


def plot_sparse_gp_inducing_points(optimizer: Foretuner, trials: List[Trial]):
    """Visualize SparseGP inducing points if available"""
    if not hasattr(optimizer.model, "get_inducing_points"):
        print("⚠️ Model doesn't have inducing points to visualize")
        return

    inducing_points = optimizer.model.get_inducing_points()
    if inducing_points is None:
        print("⚠️ No inducing points available")
        return

    # Get trial data
    X = np.array([optimizer._param_converter.to_array(t.params) for t in trials])
    y = np.array([t.value for t in trials])

    # Only visualize if 2D or can project to 2D
    n_dims = X.shape[1]
    if n_dims == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot trials
        scatter = ax.scatter(
            X[:, 0],
            y,
            c=y,
            cmap="viridis_r",
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
            label="Trials",
        )

        # Plot inducing points
        inducing_y = np.interp(
            inducing_points[:, 0],
            sorted(range(len(y)), key=lambda i: X[i, 0]),
            sorted(y),
        )
        ax.scatter(
            inducing_points[:, 0],
            inducing_y,
            c="red",
            s=100,
            marker="^",
            edgecolors="black",
            linewidth=2,
            label=f"Inducing Points ({len(inducing_points)})",
        )

        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Objective Value")
        ax.set_title("SparseGP Inducing Points (1D)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)

    elif n_dims >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 2D parameter space
        ax = axes[0]
        scatter = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap="viridis_r",
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
            label="Trials",
        )
        ax.scatter(
            inducing_points[:, 0],
            inducing_points[:, 1],
            c="red",
            s=100,
            marker="^",
            edgecolors="black",
            linewidth=2,
            label=f"Inducing Points ({len(inducing_points)})",
        )

        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")
        ax.set_title("SparseGP Inducing Points (Parameter Space)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)

        # Distribution of inducing points vs trials
        ax = axes[1]
        ax.hist2d(
            X[:, 0], X[:, 1], bins=20, alpha=0.6, cmap="Blues", label="Trial Density"
        )
        ax.scatter(
            inducing_points[:, 0],
            inducing_points[:, 1],
            c="red",
            s=100,
            marker="^",
            edgecolors="black",
            linewidth=2,
            label="Inducing Points",
        )

        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")
        ax.set_title("Inducing Points vs Trial Density")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Print inducing point statistics
    print(f"\n🎯 SparseGP Inducing Points Analysis:")
    print(f"   Number of inducing points: {len(inducing_points)}")
    print(f"   Coverage efficiency: {len(inducing_points) / len(trials) * 100:.1f}%")

    # Compute distances between inducing points
    if len(inducing_points) > 1:
        distances = []
        for i in range(len(inducing_points)):
            for j in range(i + 1, len(inducing_points)):
                dist = np.linalg.norm(inducing_points[i] - inducing_points[j])
                distances.append(dist)

        print(f"   Avg distance between inducing points: {np.mean(distances):.4f}")
        print(f"   Min distance: {np.min(distances):.4f}")
        print(f"   Max distance: {np.max(distances):.4f}")
