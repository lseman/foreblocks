# moe_logging_and_report.py
# ──────────────────────────────────────────────────────────────────────────────
# Minimal utilities to LOG and PLOT how a Mixture-of-Experts (MoE) helped your
# Transformer-based ForecastingModel. Designed to be dropped into any PyTorch
# project. No seaborn; only matplotlib + numpy.
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                         LOGGING: TRAIN/EVAL SIDE                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class MoELogger:
    """
    Collects per-step router statistics for later reporting.
    Typical usage:
        moe_log = MoELogger()
        ...
        with torch.no_grad():
            probs = gate_logits.softmax(-1)            # [N, E]
            moe_log.log_router(
                step=global_step,
                gate_logits=gate_logits,
                topk_idx=topk_idx,                     # [N, K]
                capacity_dropped=tokens_dropped,
                aux_loss=aux,
                latency_ms=latency,
                meta={"hour": hours_tensor, "node_id": node_ids_tensor}
            )
    After training/eval, call moe_log.to_json("moe_log.json") or pass moe_log.state_dict()
    to the plotting/report functions below.
    """

    def __init__(self, max_buffer_steps: Optional[int] = None) -> None:
        self.buff: Dict[str, List[Any]] = {}
        self.max_buffer_steps = max_buffer_steps

    def _append(self, key: str, value: Any) -> None:
        self.buff.setdefault(key, []).append(value)
        if self.max_buffer_steps is not None:
            # optionally keep a rolling window
            if len(self.buff[key]) > self.max_buffer_steps:
                self.buff[key].pop(0)

    def state_dict(self) -> Dict[str, List[Any]]:
        return self.buff

    def load_state_dict(self, d: Dict[str, List[Any]]) -> None:
        self.buff = d

    def to_json(self, path: str) -> None:
        # Convert numpy to lists where needed
        safe = {}
        for k, v in self.buff.items():
            safe[k] = []
            for item in v:
                if hasattr(item, "tolist"):
                    safe[k].append(item.tolist())
                else:
                    safe[k].append(item)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(safe, f)

    @staticmethod
    def from_json(path: str) -> "MoELogger":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        m = MoELogger()
        m.load_state_dict(d)
        return m

    # Main per-step logging entry point
    def log_router(
        self,
        step: Union[int, float],
        gate_logits: Union["np.ndarray", "torch.Tensor"],  # [N, E]
        topk_idx: Union["np.ndarray", "torch.Tensor"],     # [N, K]
        capacity_dropped: int = 0,
        aux_loss: Optional[float] = None,
        latency_ms: Optional[float] = None,
        meta: Optional[Dict[str, Union["np.ndarray", "torch.Tensor", Sequence[int], Sequence[float]]]] = None,
    ) -> None:
        # convert to numpy
        gate_logits = _to_numpy(gate_logits)
        topk_idx = _to_numpy(topk_idx).astype(np.int64)
        probs = _softmax(gate_logits, axis=-1)  # [N, E]

        # Router entropy (avg over tokens)
        router_entropy = float(np.mean(_entropy(probs, axis=-1)))

        # Utilization distribution across experts
        E = probs.shape[-1]
        counts = np.bincount(topk_idx.reshape(-1), minlength=E).astype(np.float64)
        util = counts / (counts.sum() + 1e-12)  # [E]

        self._append("step", float(step))
        self._append("router_entropy", router_entropy)
        self._append("expert_util", util.tolist())
        self._append("tokens_dropped", int(capacity_dropped))
        if aux_loss is not None: self._append("aux_loss", float(aux_loss))
        if latency_ms is not None: self._append("latency_ms", float(latency_ms))

        # Optionally store raw routing for later specialization plots (keep light)
        # We store downsampled snapshots to limit memory (customize as needed)
        if len(self.buff.get("snap_topk", [])) < 64:
            self._append("snap_topk", topk_idx[:4096].copy())  # snapshot
            # also store top-1 confidence for calibration plots
            top1 = np.take_along_axis(probs, topk_idx[:, :1], axis=-1).squeeze(-1)  # [N]
            self._append("snap_top1_conf", top1[:4096].copy())

        # Handle meta (any conditioning like hour, node_id, horizon, feature_id)
        if meta:
            packed = {}
            for k, v in meta.items():
                arr = _to_numpy(v)
                # try to snapshot aligned with snap_topk when possible
                if len(self.buff.get("snap_meta_" + k, [])) < 64:
                    packed[k] = arr[:4096].copy()
                    self._append("snap_meta_" + k, packed[k])


# Optional helper to attach a forward hook to your router module
def attach_router_hook(module, moe_logger: MoELogger, step_getter):
    """
    Example usage:
        hook = attach_router_hook(model.encoder.layers[0].moe.router, moe_log, lambda: global_step)
        ...
        hook.remove()
    Expects the router module forward(...) to produce gate_logits and topk_idx somewhere.
    Customize extraction below to match your implementation.
    """
    import torch

    def _hook(mod, inp, out):
        # Customize this based on your router outputs
        # For example, if out is a tuple: (expert_outputs, gate_logits, topk_idx, tokens_dropped, aux_loss)
        gate_logits = getattr(mod, "last_gate_logits", None)
        topk_idx = getattr(mod, "last_topk_idx", None)
        tokens_dropped = int(getattr(mod, "last_tokens_dropped", 0))
        aux_loss = float(getattr(mod, "last_aux_loss", 0.0))

        if gate_logits is None or topk_idx is None:
            return

        moe_logger.log_router(
            step=step_getter(),
            gate_logits=gate_logits,
            topk_idx=topk_idx,
            capacity_dropped=tokens_dropped,
            aux_loss=aux_loss,
            latency_ms=float(getattr(mod, "last_latency_ms", 0.0)),
            meta=getattr(mod, "last_meta", None)
        )

    return module.register_forward_hook(_hook)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                              PLOTTING SUITE                             ║
# ╚═════════════════════════════════════════════════════════════════════════╝

# ——— Utilities ————————————————————————————————————————————————

def _to_numpy(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def _entropy(p: np.ndarray, axis=-1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)

def _steps(d: Dict[str, List[Any]]) -> np.ndarray:
    return np.asarray(d.get("step", []), dtype=float)

def _stack_list_of_lists(x: List[List[float]]) -> np.ndarray:
    return np.asarray(x, dtype=float) if len(x) > 0 else np.zeros((0,))

def _maybe_save(fig, outdir: Optional[str], name: str):
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, name), bbox_inches="tight")


# ——— 1) Expert Utilization over Time ————————————————————————————

def plot_expert_utilization_over_time(log: Dict[str, List[Any]], title="Expert Utilization"):
    steps = _steps(log)
    util = np.asarray(log.get("expert_util", []), dtype=float)   # [S, E]
    if util.ndim != 2 or util.shape[0] == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(steps, util.T, step='mid')
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of routed tokens")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

# ——— 2) Load Imbalance CV ——————————————————————————————————————

def plot_load_imbalance_cv(log: Dict[str, List[Any]], title="Expert Load Imbalance (CV)"):
    util = np.asarray(log.get("expert_util", []), dtype=float)  # [S, E]
    steps = _steps(log)
    if util.ndim != 2 or util.shape[0] == 0:
        return None
    mean = util.mean(axis=1) + 1e-12
    cv = util.std(axis=1) / mean
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(steps, cv)
    ax.set_xlabel("Step"); ax.set_ylabel("CV ↓")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

# ——— 3) Router Entropy & Aux Loss ——————————————————————————————

def plot_entropy_and_aux(log: Dict[str, List[Any]]):
    steps = _steps(log)
    ent = _stack_list_of_lists(log.get("router_entropy", []))
    aux = _stack_list_of_lists(log.get("aux_loss", []))
    fig, ax = plt.subplots(figsize=(9, 3))
    if len(ent) > 0:
        ax.plot(steps[:len(ent)], ent, label="Router Entropy")
    if len(aux) > 0:
        ax.plot(steps[:len(aux)], aux, label="Aux Loss")
    ax.set_xlabel("Step"); ax.set_title("Router Entropy / Aux Loss")
    ax.grid(True, alpha=0.3); ax.legend()
    return fig

# ——— 4) Tokens Dropped (Capacity Pressure) ————————————————————

def plot_tokens_dropped(log: Dict[str, List[Any]]):
    steps = _steps(log)
    dropped = np.asarray(log.get("tokens_dropped", []), dtype=float)
    if dropped.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(steps[:len(dropped)], dropped)
    ax.set_xlabel("Step"); ax.set_ylabel("# Tokens")
    ax.set_title("Tokens Dropped (capacity overflow)")
    ax.grid(True, alpha=0.3)
    return fig

# ——— 5) Specialization Heatmap (expert × condition) ———————————

def heatmap_expert_by_condition(
    topk_snapshots: List[np.ndarray],   # a few [N,K] arrays
    cond_snapshots: List[np.ndarray],   # a few [N] arrays in [0,C)
    E: int,
    C: int,
    title: str = "Expert × Condition usage"
):
    counts = np.zeros((E, C), dtype=np.int64)
    for topk, cond in zip(topk_snapshots, cond_snapshots):
        topk = np.asarray(topk).reshape(-1)      # use primary expert only
        cond = np.asarray(cond).reshape(-1)
        m = min(len(topk), len(cond))
        if m <= 0: continue
        for e, c in zip(topk[:m], cond[:m]):
            if 0 <= e < E and 0 <= c < C:
                counts[e, c] += 1
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(counts, aspect='auto', origin='lower')
    ax.set_ylabel("Expert"); ax.set_xlabel("Condition bin")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9)
    return fig, counts

# ——— 6) Router Reliability Diagram (Calibration proxy) —————————

def reliability_diagram(
    conf: np.ndarray,
    correctness: np.ndarray,
    bins: int = 10,
    title: str = "Router Reliability"
):
    conf = np.asarray(conf).astype(float)
    correctness = np.asarray(correctness).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    out_conf, out_acc = [], []
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.any():
            out_conf.append(conf[m].mean())
            out_acc.append(correctness[m].mean())
        else:
            out_conf.append(np.nan)
            out_acc.append(np.nan)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], '--', alpha=0.5)
    ax.plot(out_conf, out_acc, marker='o')
    ax.set_xlabel("Confidence"); ax.set_ylabel("Observed accuracy (proxy)")
    ax.set_title(title); ax.grid(True, alpha=0.3)
    return fig

# ——— 7) Expert Ablation (Δ metric per expert) ————————————————

def bar_ablation_delta(metric_full: float, metric_without_expert: Sequence[float], title="Per-Expert Importance (Ablation)"):
    metric_without_expert = np.asarray(metric_without_expert, dtype=float)
    delta = metric_without_expert - metric_full
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(np.arange(len(delta)), delta)
    ax.axhline(0, lw=1, color='k')
    ax.set_xlabel("Expert"); ax.set_ylabel("Δ Metric (↑ worse)")
    ax.set_title(title)
    return fig

# ——— 8) Horizon-wise Error (MoE vs Dense) ————————————————

def plot_horizon_wise_error(err_moe: Sequence[float], err_dense: Sequence[float], title="Horizon-wise Error"):
    err_moe = np.asarray(err_moe, dtype=float)
    err_dense = np.asarray(err_dense, dtype=float)
    H = np.arange(len(err_moe))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(H, err_moe, label="MoE")
    ax.plot(H, err_dense, label="Dense")
    ax.set_xlabel("Horizon τ"); ax.set_ylabel("Error"); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend()
    return fig

# ——— 9) Accuracy–Latency Pareto ——————————————————————————————

def pareto_accuracy_latency(points: List[Dict[str, float]], title="Accuracy–Latency Pareto"):
    """
    points: [{"name": "MoE", "metric": 0.13, "latency_ms": 7.5, "size_m": 120}, ...]
    Lower metric is better (e.g., sMAPE, MASE, RMSE).
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for p in points:
        size = 20 + 2 * float(p.get("size_m", 10))
        ax.scatter(p["latency_ms"], p["metric"], s=size)
        ax.annotate(p["name"], (p["latency_ms"], p["metric"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Latency (ms) ↓"); ax.set_ylabel("Validation Error ↓"); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig

# ——— 10) Attention × Expert Interaction ———————————————————————

def plot_attn_entropy_by_expert(attn_entropy_per_token: np.ndarray, primary_expert_per_token: np.ndarray, E: int, title="Attention Entropy by Expert"):
    """
    attn_entropy_per_token: [N] entropy values computed per token from your attention maps
    primary_expert_per_token: [N] expert index for that token
    """
    attn_entropy_per_token = np.asarray(attn_entropy_per_token, dtype=float)
    primary_expert_per_token = np.asarray(primary_expert_per_token, dtype=int)
    vals = [attn_entropy_per_token[primary_expert_per_token == e] for e in range(E)]
    means = [float(v.mean()) if v.size > 0 else np.nan for v in vals]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(E), means)
    ax.set_xlabel("Expert"); ax.set_ylabel("Mean attention entropy")
    ax.set_title(title); ax.grid(True, axis='y', alpha=0.2)
    return fig


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                         REPORT / EXPORT CONVENIENCE                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

@dataclass
class ReportInputs:
    log: Dict[str, List[Any]]
    # Optional extras for specific plots:
    # For heatmap:
    E: Optional[int] = None
    condition_name: Optional[str] = None  # e.g., "hour", "node_id", "horizon"
    condition_cardinality: Optional[int] = None  # e.g., 24 for hour
    # For reliability:
    reliability_conf: Optional[np.ndarray] = None
    reliability_correct: Optional[np.ndarray] = None
    # For ablation:
    full_metric: Optional[float] = None
    metric_without_expert: Optional[Sequence[float]] = None
    # For horizon-wise:
    err_moe: Optional[Sequence[float]] = None
    err_dense: Optional[Sequence[float]] = None
    # For pareto:
    pareto_points: Optional[List[Dict[str, float]]] = None
    # For attention × expert:
    attn_entropy_per_token: Optional[np.ndarray] = None
    primary_expert_per_token: Optional[np.ndarray] = None

def _maybe_save(fig, outdir: str | None, name: str, close: bool = True) -> str | None:
    """
    Save a matplotlib Figure if it's not None.
    Returns the saved path or None when there's nothing to save.
    """
    if fig is None:
        return None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, name)
        fig.savefig(path, bbox_inches="tight")
        if close:
            plt.close(fig)
        return path
    # If no outdir, we still return a synthetic name to indicate it was created
    return name

def build_moe_report(
    ri: ReportInputs,
    outdir: Optional[str] = None,
) -> Dict[str, Optional[plt.Figure]]:
    """
    Builds and (optionally) saves all core figures. Returns dict of figures.
    Also prints a short manifest of what was saved or skipped.
    """
    figs: Dict[str, Optional[plt.Figure]] = {}
    manifest: Dict[str, str] = {}

    # 1 Utilization over time
    figs["util_over_time"] = plot_expert_utilization_over_time(ri.log)
    p = _maybe_save(figs["util_over_time"], outdir, "1_utilization_over_time.png")
    manifest["util_over_time"] = p or "skipped (missing expert_util over steps)"

    # 2 Load imbalance CV
    figs["imbalance_cv"] = plot_load_imbalance_cv(ri.log)
    p = _maybe_save(figs["imbalance_cv"], outdir, "2_imbalance_cv.png")
    manifest["imbalance_cv"] = p or "skipped (missing/degenerate expert_util)"

    # 3 Entropy & Aux
    figs["entropy_aux"] = plot_entropy_and_aux(ri.log)
    p = _maybe_save(figs["entropy_aux"], outdir, "3_entropy_aux.png")
    manifest["entropy_aux"] = p or "skipped (no router_entropy/aux_loss)"

    # 4 Tokens dropped
    figs["tokens_dropped"] = plot_tokens_dropped(ri.log)
    p = _maybe_save(figs["tokens_dropped"], outdir, "4_tokens_dropped.png")
    manifest["tokens_dropped"] = p or "skipped (no tokens_dropped)"

    # 5 Specialization heatmap (if we have snapshots + condition)
    figs["heatmap_specialization"] = None
    if ri.E is not None and ri.condition_name and ri.condition_cardinality:
        topk_snaps = ri.log.get("snap_topk", [])
        cond_snaps = ri.log.get("snap_meta_" + ri.condition_name, [])
        if len(topk_snaps) > 0 and len(cond_snaps) > 0:
            fig, _ = heatmap_expert_by_condition(
                topk_snapshots=[_to_numpy(x) for x in topk_snaps],
                cond_snapshots=[_to_numpy(x) for x in cond_snaps],
                E=ri.E, C=ri.condition_cardinality,
                title=f"Expert × {ri.condition_name}"
            )
            figs["heatmap_specialization"] = fig
            p = _maybe_save(fig, outdir, f"5_heatmap_expert_by_{ri.condition_name}.png")
            manifest["heatmap_specialization"] = p or "skipped (unexpected)"
        else:
            manifest["heatmap_specialization"] = "skipped (missing snap_topk or snap_meta)"
    else:
        manifest["heatmap_specialization"] = "skipped (E/condition not provided)"

    # 6 Reliability
    figs["reliability"] = None
    if ri.reliability_conf is not None and ri.reliability_correct is not None:
        figs["reliability"] = reliability_diagram(
            conf=ri.reliability_conf, correctness=ri.reliability_correct
        )
        p = _maybe_save(figs["reliability"], outdir, "6_router_reliability.png")
        manifest["reliability"] = p or "skipped (unexpected)"
    else:
        manifest["reliability"] = "skipped (no reliability_conf/correct)"

    # 7 Ablation
    figs["ablation"] = None
    if ri.full_metric is not None and ri.metric_without_expert is not None:
        figs["ablation"] = bar_ablation_delta(
            metric_full=ri.full_metric,
            metric_without_expert=ri.metric_without_expert
        )
        p = _maybe_save(figs["ablation"], outdir, "7_expert_ablation.png")
        manifest["ablation"] = p or "skipped (unexpected)"
    else:
        manifest["ablation"] = "skipped (no ablation inputs)"

    # 8 Horizon-wise
    figs["horizon_error"] = None
    if ri.err_moe is not None and ri.err_dense is not None:
        figs["horizon_error"] = plot_horizon_wise_error(
            err_moe=ri.err_moe, err_dense=ri.err_dense
        )
        p = _maybe_save(figs["horizon_error"], outdir, "8_horizon_wise_error.png")
        manifest["horizon_error"] = p or "skipped (unexpected)"
    else:
        manifest["horizon_error"] = "skipped (no horizon curves)"

    # 9 Pareto
    figs["pareto"] = None
    if ri.pareto_points is not None:
        figs["pareto"] = pareto_accuracy_latency(ri.pareto_points)
        p = _maybe_save(figs["pareto"], outdir, "9_accuracy_latency_pareto.png")
        manifest["pareto"] = p or "skipped (unexpected)"
    else:
        manifest["pareto"] = "skipped (no pareto_points)"

    # 10 Attention × Expert
    figs["attn_by_expert"] = None
    if ri.attn_entropy_per_token is not None and ri.primary_expert_per_token is not None and ri.E is not None:
        figs["attn_by_expert"] = plot_attn_entropy_by_expert(
            ri.attn_entropy_per_token, ri.primary_expert_per_token, E=ri.E
        )
        p = _maybe_save(figs["attn_by_expert"], outdir, "10_attention_entropy_by_expert.png")
        manifest["attn_by_expert"] = p or "skipped (unexpected)"
    else:
        manifest["attn_by_expert"] = "skipped (no attention/primary expert inputs)"

    # Optional: print a compact manifest so you see why things were skipped
    if outdir:
        print("MoE report manifest:")
        for k, v in manifest.items():
            print(f"  - {k}: {v}")

    return figs

# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                    EVALUATION-TIME HELPER (ABLATION)                    ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def run_per_expert_ablation(
    eval_fn_disable_expert,   # callable: (e:int) -> metric_value (lower is better)
    E: int
) -> List[float]:
    """
    Convenience wrapper to measure metric when each expert is disabled.
    Example:
        def eval_fn_disable_expert(e):
            return evaluate_model(disable_expert=e)  # returns sMAPE or RMSE
        results = run_per_expert_ablation(eval_fn_disable_expert, E)
    """
    out = []
    for e in range(E):
        out.append(float(eval_fn_disable_expert(e)))
    return out


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                     RELIABILITY PROXY CONSTRUCTION IDEA                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def make_reliability_proxy_from_contrib(
    top1_conf_list: List[np.ndarray],
    delta_improvement_list: List[np.ndarray],
    threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (confidence, correctness) arrays for reliability_diagram:
    - top1_conf_list: list of arrays with top-1 gate probs per token.
    - delta_improvement_list: list of arrays with improvement vs. a baseline
      (e.g., per-token negative delta in loss when using routed expert vs. average).
    - correctness: 1 if improvement >= threshold.
    Returns flat arrays (conf, correctness).
    """
    conf = np.concatenate([np.asarray(c).reshape(-1) for c in top1_conf_list], axis=0)
    imp = np.concatenate([np.asarray(d).reshape(-1) for d in delta_improvement_list], axis=0)
    corr = (imp >= threshold).astype(float)
    return conf, corr


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                      SIMPLE SELF-CHECK (OPTIONAL DEMO)                  ║
# ╚═════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    # Generate a tiny synthetic log to test all plots quickly.
    rng = np.random.default_rng(0)
    S, E, K = 50, 6, 2
    steps = np.arange(S) * 100

    log = {
        "step": steps.tolist(),
        "expert_util": [],
        "router_entropy": [],
        "tokens_dropped": [],
        "aux_loss": [],
        "latency_ms": [],
        "snap_topk": [],
        "snap_top1_conf": [],
        "snap_meta_hour": [],   # pretend our condition is 'hour'
    }

    for s in range(S):
        # make a utilization distribution that evolves
        raw = rng.random(E) + np.linspace(0, 1, E) * (s / S) * 0.1
        util = raw / raw.sum()
        log["expert_util"].append(util.tolist())

        # fake entropy and aux loss trends
        log["router_entropy"].append(float(1.2 - 0.5 * (s / S)))
        log["aux_loss"].append(float(0.2 + 0.3 * (s / S)))
        log["tokens_dropped"].append(int(rng.poisson(5 * (s / S))))
        log["latency_ms"].append(float(7.0 + rng.normal(0, 0.2)))

        # snapshots
        N = 200
        probs = rng.random((N, E)); probs = probs / probs.sum(-1, keepdims=True)
        topk = np.argsort(-probs, axis=-1)[:, :K]
        log["snap_topk"].append(topk)
        log["snap_top1_conf"].append(probs[np.arange(N), topk[:, 0]])
        log["snap_meta_hour"].append(rng.integers(0, 24, size=(N,)))

    # Reliability proxy
    conf, corr = make_reliability_proxy_from_contrib(
        top1_conf_list=log["snap_top1_conf"],
        delta_improvement_list=[rng.normal(0.05, 0.1, size=len(x)) for x in log["snap_top1_conf"]],
        threshold=0.0
    )

    # Build report
    ri = ReportInputs(
        log=log,
        E=E,
        condition_name="hour",
        condition_cardinality=24,
        reliability_conf=conf,
        reliability_correct=corr,
        full_metric=0.120,  # e.g., sMAPE
        metric_without_expert=[0.122, 0.130, 0.121, 0.140, 0.128, 0.124],
        err_moe=[0.11, 0.12, 0.13, 0.15, 0.18],
        err_dense=[0.12, 0.13, 0.15, 0.19, 0.24],
        pareto_points=[
            {"name": "Dense", "metric": 0.140, "latency_ms": 6.0, "size_m": 80},
            {"name": "MoE-4e", "metric": 0.125, "latency_ms": 6.8, "size_m": 95},
            {"name": "MoE-6e", "metric": 0.120, "latency_ms": 7.2, "size_m": 120},
        ],
        attn_entropy_per_token=np.abs(rng.normal(1.0, 0.25, size=(500,))),
        primary_expert_per_token=rng.integers(0, E, size=(500,))
    )

    figs = build_moe_report(ri, outdir="moe_report_example")
    print("Saved example report figures to ./moe_report_example")
