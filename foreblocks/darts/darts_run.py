# ─── Standard Library ─────────────────────────────────────────────
import contextlib
import copy
import random
import re
import time
import types
from collections import defaultdict

# ─── Third-Party Libraries ────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # (only include if used later)
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from darts import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# Context manager to disable CuDNN temporarily
@contextlib.contextmanager
def disable_cudnn():
    """
    Context manager to temporarily disable CuDNN for RNN operations
    """
    prev_flag = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = prev_flag


def compute_jacob_cov(model, inputs, n_samples=10):
    """
    Compute Jacobian Covariance score - measures the diversity of features
    With CuDNN handling for RNNs and Transformers
    """
    try:
        # Always force model to training mode for RNN backward pass compatibility
        was_training = model.training
        model.train()

        # Get a sample of inputs if batch is large
        if inputs.size(0) > n_samples:
            indices = torch.randperm(inputs.size(0))[:n_samples]
            inputs_sample = inputs[indices]
        else:
            inputs_sample = inputs

        batch_size = inputs_sample.size(0)

        # Ensure input requires gradient
        inputs_sample = inputs_sample.detach().clone().requires_grad_(True)

        # Disable CuDNN temporarily for backward compatibility
        with disable_cudnn():
            # Get model output
            outputs = model(inputs_sample)

        # Handle different output formats
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # For regression tasks, handle 1D outputs
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

        output_size = outputs.size(1) if outputs.dim() > 1 else 1
        input_size = inputs_sample.view(batch_size, -1).size(1)

        # Prepare container for jacobian
        jacob = torch.zeros(
            batch_size, output_size, input_size, device=inputs_sample.device
        )

        # Compute Jacobian matrix row by row
        for i in range(min(output_size, 10)):  # Limit to 10 outputs for efficiency
            model.zero_grad()
            inputs_sample.grad = None

            # Select output
            if outputs.dim() > 1:
                out = outputs[:, i].sum()
            else:
                out = outputs.sum()

            # Compute gradient
            out.backward(retain_graph=(i < output_size - 1))

            # Extract input gradient (Jacobian row)
            if inputs_sample.grad is not None:
                for j in range(batch_size):
                    jacob[j, i] = inputs_sample.grad[j].flatten()
            else:
                # If grad is None, there's a computational issue
                print("Warning: inputs_sample.grad is None")
                continue

        # Clean up
        inputs_sample.requires_grad_(False)
        model.zero_grad()

        # Compute Jacobian covariance and score
        cov_score = 0
        for j in range(batch_size):
            J = jacob[j]
            # Filter out zero rows
            mask = J.abs().sum(dim=1) > 0
            if mask.sum() > 1:  # Need at least 2 non-zero rows
                J_filtered = J[mask]
                # Compute covariance matrix
                JJ = J_filtered @ J_filtered.t()

                # Get eigenvalues
                try:
                    eigs = torch.linalg.eigvalsh(
                        JJ + 1e-6 * torch.eye(JJ.size(0), device=JJ.device)
                    )
                    # Keep positive eigenvalues
                    eigs = eigs[eigs > 1e-6]
                    if len(eigs) > 0:
                        eigs_norm = eigs / eigs.sum()
                        # Entropy as diversity measure
                        entropy = -torch.sum(eigs_norm * torch.log(eigs_norm + 1e-10))
                        cov_score += entropy.item()
                except Exception as e:
                    print(f"  SVD computation error in Jacob_Cov: {str(e)}")
                    continue

        score = cov_score / max(batch_size, 1)

        # Restore original training state
        if not was_training:
            model.eval()

        return float(score)
    except Exception as e:
        print(f"Jacob_Cov computation error details: {e}")
        # Return a default value rather than 0 to distinguish from valid computations
        return -1.0  # Use a sentinel value to indicate error


@contextlib.contextmanager
def disable_efficient_attention():
    """
    Context manager to disable PyTorch 2.0+ efficient attention implementation
    and fall back to the standard implementation that supports backward passes
    """
    # Store original functions that we'll patch
    original_sdpa = None
    original_flash = None

    # Check if we're using PyTorch 2.0+
    has_torch2_attention = hasattr(F, "scaled_dot_product_attention")

    if has_torch2_attention:
        # Save the original function
        original_sdpa = F.scaled_dot_product_attention

        # Define a replacement that uses standard attention
        def manual_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        ):
            # Standard implementation of scaled dot-product attention
            if scale is None:
                scale = 1.0 / (query.size(-1) ** 0.5)

            # Calculate attention scores
            attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale

            # Apply mask if provided
            if attn_mask is not None:
                attn_weight = attn_weight + attn_mask
            elif is_causal:
                # Create causal mask
                seq_len = query.size(-2)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_weight = attn_weight.masked_fill(causal_mask, float("-inf"))

            # Apply softmax
            attn_weight = F.softmax(attn_weight, dim=-1)

            # Apply dropout
            if dropout_p > 0.0:
                attn_weight = F.dropout(attn_weight, p=dropout_p)

            # Calculate output
            output = torch.matmul(attn_weight, value)
            return output

        # Replace the function
        F.scaled_dot_product_attention = manual_sdpa

    # Check if torch._C._nn has flash attention
    has_flash_attention = hasattr(torch._C._nn, "flash_attention")
    if has_flash_attention:
        # Save the original function
        original_flash = torch._C._nn.flash_attention

        # Replace it with a function that raises an error
        def disabled_flash(*args, **kwargs):
            raise NotImplementedError("Flash attention has been temporarily disabled")

        torch._C._nn.flash_attention = disabled_flash

    try:
        yield
    finally:
        # Restore original functions
        if has_torch2_attention and original_sdpa is not None:
            F.scaled_dot_product_attention = original_sdpa

        if has_flash_attention and original_flash is not None:
            torch._C._nn.flash_attention = original_flash


def compute_grasp_transformer_aware(model, inputs, targets):
    """
    GraSP metric (Gradient Signal Preservation) with special handling for transformers
    that use scaled dot-product attention
    """
    try:
        # Save model's training state
        was_training = model.training
        model.train()  # Set to train mode for gradient computation

        # Always make copies of inputs/targets to avoid modifying originals
        inputs = inputs.clone().detach()
        targets = targets.clone().detach()

        # Handle data type properly based on task
        if targets.dtype == torch.long:
            # Classification task - use CrossEntropyLoss
            criterion = nn.CrossEntropyLoss()
            # Make sure targets are long type
            targets = targets.long()
        else:
            # Regression task - use MSELoss
            criterion = nn.MSELoss()
            # Make sure targets are float type
            targets = targets.float()

        # Find all attention modules to handle them specially
        attention_modules = []

        def find_attention_modules(module, prefix=""):
            for name, child in module.named_children():
                child_path = f"{prefix}.{name}" if prefix else name
                # Look for attention modules by name pattern
                if re.search(r"(attention|attn|self_attn)", name.lower()):
                    attention_modules.append((child_path, child))
                # Recursively search in child modules
                find_attention_modules(child, child_path)

        find_attention_modules(model)

        # Disable efficient attention implementation
        with disable_efficient_attention():
            # First forward pass
            outputs = model(inputs)

            # Handle potential shape mismatches depending on task type
            if targets.dtype == torch.long:
                # For classification - ensure outputs and targets have correct shapes
                if outputs.dim() > 2:
                    if outputs.dim() == 3:  # [batch, seq_len, num_classes]
                        # For sequence models, use the last token prediction
                        outputs = outputs[:, -1]
                    else:
                        outputs = outputs.view(outputs.size(0), -1)

                if targets.dim() > 1 and targets.size(1) == 1:
                    targets = targets.squeeze(1)

                loss = criterion(outputs, targets)
            else:
                # For regression - ensure outputs match target shape
                if outputs.shape != targets.shape:
                    # Try to reshape outputs to match targets
                    if outputs.dim() == 1 and targets.dim() == 2:
                        outputs = outputs.unsqueeze(1)
                    elif outputs.dim() == 2 and targets.dim() == 1:
                        targets = targets.unsqueeze(1)
                    elif outputs.dim() == 3:  # [batch, seq_len, features]
                        # Use last prediction or mean
                        outputs = outputs[:, -1]

                loss = criterion(outputs, targets)

            # Use a loop to accumulate gradients without using create_graph
            # This avoids double backward which can cause issues
            all_grads = []

            # First pass - get first-order gradients
            model.zero_grad()
            loss.backward(retain_graph=True)

            # Store gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Store a copy of the gradients
                    all_grads.append((name, param.grad.clone()))

            # Compute GraSP score directly without second backward
            score = 0
            for name, grad in all_grads:
                param = dict(model.named_parameters())[name]
                # Use parameter values and gradients to approximate score
                score += (param * grad).sum().item()

        model.zero_grad()

        # Restore original training state
        if not was_training:
            model.eval()

        return float(abs(score))
    except Exception as e:
        print(f"GraSP computation error details: {e}")
        return -1.0  # Use sentinel value to indicate error


def is_transformer_model(model):
    """
    Detect if a model is transformer-based by looking for attention modules
    """
    # Common transformer module names
    transformer_patterns = [
        r"(self_attention|mha|multihead|attention|attn)",
        r"(transformer|mhsa)",
        r"(cross_attention)",
    ]

    # Check model structure
    found = False
    for name, _ in model.named_modules():
        for pattern in transformer_patterns:
            if re.search(pattern, name.lower()):
                found = True
                break
        if found:
            break

    return found


def compute_grasp_robust(model, inputs, targets):
    """
    Wrapper function that chooses the appropriate GraSP implementation
    based on model architecture
    """
    # Check if we're dealing with a transformer-based model
    if is_transformer_model(model):
        print("Detected transformer model - using specialized GraSP implementation")
        return compute_grasp_transformer_aware(model, inputs, targets)
    else:
        # For non-transformer models, use the standard implementation
        # but with CuDNN disabled
        with disable_cudnn():
            try:
                return compute_grasp(model, inputs, targets)
            except Exception as e:
                print(f"Standard GraSP failed, falling back to transformer-aware: {e}")
                return compute_grasp_transformer_aware(model, inputs, targets)


# Original compute_grasp and disable_cudnn from previous code
@contextlib.contextmanager
def disable_cudnn():
    """
    Context manager to temporarily disable CuDNN for RNN operations
    """
    prev_flag = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = prev_flag


def compute_grasp(model, inputs, targets):
    """
    GraSP metric (Gradient Signal Preservation)
    With CuDNN handling for RNNs
    """
    try:
        # Save model's training state
        was_training = model.training
        model.train()  # Set to train mode for gradient computation

        # Always make copies of inputs/targets to avoid modifying originals
        inputs = inputs.clone().detach()
        targets = targets.clone().detach()

        # Handle data type properly based on task
        if targets.dtype == torch.long:
            # Classification task - use CrossEntropyLoss
            criterion = nn.CrossEntropyLoss()
            # Make sure targets are long type
            targets = targets.long()
        else:
            # Regression task - use MSELoss
            criterion = nn.MSELoss()
            # Make sure targets are float type
            targets = targets.float()

        # First forward pass
        outputs = model(inputs)

        # Handle potential shape mismatches depending on task type
        if targets.dtype == torch.long:
            # For classification - ensure outputs and targets have correct shapes
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            if targets.dim() > 1 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            loss = criterion(outputs, targets)
        else:
            # For regression - ensure outputs match target shape
            if outputs.shape != targets.shape:
                # Try to reshape outputs to match targets
                if outputs.dim() == 1 and targets.dim() == 2:
                    outputs = outputs.unsqueeze(1)
                elif outputs.dim() == 2 and targets.dim() == 1:
                    targets = targets.unsqueeze(1)
            loss = criterion(outputs, targets)

        # Compute gradients with allow_unused=True to handle unused parameters
        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=True,
            allow_unused=True,
        )

        # Filter out None values from unused parameters
        grads = [g for g in grads if g is not None]

        if not grads:
            print("No gradients available - check model setup")
            return 0.0

        # Compute sum of gradients squared (gradient magnitude)
        grad_square = 0
        for grad in grads:
            grad_square += grad.pow(2).sum()

        # Second backward pass to get Hessian information
        grad_square.backward()

        # GraSP score combines gradient and Hessian information
        score = 0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                score += (p * p.grad).sum().item()

        model.zero_grad()

        # Restore original training state
        if not was_training:
            model.eval()

        return float(abs(score))
    except Exception as e:
        print(f"GraSP computation error details: {e}")
        return -1.0  # Use sentinel value to indicate error


def compute_fisher(model, inputs, targets):
    """
    Compute Fisher Information score
    With fixes for 3D outputs and CuDNN handling for RNNs
    """
    try:
        # Save model's training state
        was_training = model.training
        model.eval()  # Set to eval mode for consistent behavior

        # Ensure inputs and targets have the right types
        inputs = inputs.clone().detach()
        targets = targets.clone().detach()

        # Get model outputs with CuDNN disabled for RNNs
        with disable_cudnn():
            outputs = model(inputs)

        # Handle different output formats
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Handle 3D outputs (sequence models often output [batch_size, seq_len, features])
        if outputs.dim() == 3:
            # Take average over sequence dimension or last prediction
            outputs = outputs[:, -1, :]  # Use last timestep
            # Alternatively: outputs = outputs.mean(dim=1)  # Average over sequence

        # Create proper probability distribution based on task type
        if targets.dtype == torch.long:
            # Classification task - use softmax to get probabilities
            if outputs.dim() <= 1:
                # If 1D output, assume binary classification and use sigmoid
                prob_dist = torch.sigmoid(outputs).unsqueeze(1)
                prob_dist = torch.cat([1 - prob_dist, prob_dist], dim=1)
            else:
                # Multi-class classification
                prob_dist = F.softmax(outputs, dim=1)
        else:
            # Regression task - convert to probability distribution
            mean = outputs
            if mean.dim() == 1:
                mean = mean.unsqueeze(1)  # Ensure 2D

            # Create 2D distribution (mean, variance)
            prob_dist = torch.cat([mean, torch.ones_like(mean) * 0.01], dim=1)

        # Verification step
        if prob_dist.dim() not in [1, 2]:
            raise ValueError(f"prob_dist must be 1 or 2 dim, got {prob_dist.dim()}")

        # Compute log probabilities
        if prob_dist.dim() == 1:
            log_probs = torch.log(prob_dist + 1e-8)
        else:
            # Use negative log likelihood for 2D distribution
            if targets.dim() == 1 and prob_dist.dim() == 2:
                target_indices = targets.long().view(-1, 1)
                if target_indices.max() < prob_dist.size(1):
                    probs = torch.gather(prob_dist, 1, target_indices)
                else:
                    # If targets don't match distribution shape, use mean prob
                    probs = prob_dist.mean(dim=1, keepdim=True)
            else:
                # For regression, use all probabilities
                probs = prob_dist.mean(dim=1, keepdim=True)

            log_probs = torch.log(probs + 1e-8).squeeze()

        # Compute Fisher Information score
        fisher_score = 0
        batch_size = inputs.size(0)

        model.train()  # Temporarily set to train for gradient computation

        # Disable CuDNN for backward compatibility
        with disable_cudnn():
            # Iterate through examples in batch
            for i in range(min(batch_size, 10)):  # Limit to 10 examples for efficiency
                model.zero_grad()
                # Get gradients for this example
                if log_probs.dim() == 0:  # Single scalar
                    log_prob = log_probs
                else:
                    log_prob = log_probs[i]

                log_prob.backward(retain_graph=(i < batch_size - 1))

                # Accumulate Fisher Information
                for p in model.parameters():
                    if p.grad is not None:
                        fisher_score += p.grad.pow(2).sum().item()

        # Normalize by batch size
        fisher_score /= min(batch_size, 10)

        model.zero_grad()

        # Restore original training state
        if not was_training:
            model.eval()

        return float(fisher_score)
    except Exception as e:
        print(f"[Fisher] Error: {e}")
        return -1.0  # Use sentinel value to indicate error


def evaluate_zero_cost_metrics(
    model,
    dataloader,
    device,
    num_batches=3,
    batch_size=4,
    dataset_name=None,
    custom_weights=None,
):
    """
    Memory-efficient evaluation of zero-cost NAS metrics with improved reliability.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader containing the dataset
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate for improved stability
        batch_size: Batch size to use for each evaluation
        dataset_name: Name of dataset for using preset weights
        custom_weights: Optional dictionary of custom weights for metrics

    Returns:
        Dictionary of metrics and aggregate score
    """
    # Get multiple batches for more reliable evaluation
    batches = []
    dataloader_iter = iter(dataloader)
    for _ in range(min(num_batches, len(dataloader))):
        try:
            batch = next(dataloader_iter)
            batches.append(batch)
        except StopIteration:
            break

    if not batches:
        raise ValueError("No batches available in dataloader")

    # Track metric computation time
    timing = {}

    # Compute metrics across multiple batches
    metrics_list = []
    for batch_idx, batch in enumerate(batches):
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                batch_x, batch_y = batch[0], batch[1]
            else:
                batch_x = batch[0]
                batch_y = torch.zeros(
                    batch_x.size(0), device=device
                )  # Dummy labels if not provided
        else:
            batch_x = batch
            batch_y = torch.zeros(
                batch_x.size(0), device=device
            )  # Dummy labels if not provided

        # Use subset of batch for efficiency
        batch_x = batch_x[:batch_size].to(device)
        batch_y = batch_y[:batch_size].to(device)

        print(f"Computing metrics for batch {batch_idx+1}/{len(batches)}")
        batch_metrics = compute_batch_metrics(model, batch_x, batch_y, timing)
        metrics_list.append(batch_metrics)

    # Aggregate metrics across batches
    aggregated_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        if values:
            # Use median for robustness against outliers
            aggregated_metrics[key] = float(np.median(values))

    # Print timing information
    print("\nMetric computation times:")
    for metric, time_taken in timing.items():
        avg_time = time_taken / num_batches
        print(f"  {metric}: {avg_time:.4f} seconds per batch")

    # Add aggregate score
    agg_score = aggregate_metrics(aggregated_metrics, custom_weights, dataset_name)
    aggregated_metrics["aggregate_score"] = agg_score

    return aggregated_metrics


def compute_batch_metrics(model, inputs, targets, timing):
    """Compute all metrics for a single batch."""
    metrics = {}

    # === METRIC COMPUTATION WITH EXCEPTION HANDLING AND TIMING ===

    # SYNFLOW
    start_time = time.time()
    try:
        metrics["synflow"] = compute_synflow(model, inputs)
    except Exception as e:
        print(f"[Synflow] Error: {e}")
        metrics["synflow"] = 0.0
    timing["synflow"] = timing.get("synflow", 0) + (time.time() - start_time)

    # PARAMETER COUNT
    start_time = time.time()
    try:
        metrics["param_count"] = count_params(model)
    except Exception as e:
        print(f"[Param Count] Error: {e}")
        metrics["param_count"] = 0.0
    timing["param_count"] = timing.get("param_count", 0) + (time.time() - start_time)

    # NASWOT
    start_time = time.time()
    try:
        metrics["naswot"] = compute_naswot(model, inputs)
    except Exception as e:
        print(f"[NASWOT] Error: {e}")
        metrics["naswot"] = 0.0
    timing["naswot"] = timing.get("naswot", 0) + (time.time() - start_time)

    # SENSITIVITY
    start_time = time.time()
    try:
        metrics["sensitivity"] = compute_sensitivity(model, inputs)
    except Exception as e:
        print(f"[Sensitivity] Error: {e}")
        metrics["sensitivity"] = 0.0
    timing["sensitivity"] = timing.get("sensitivity", 0) + (time.time() - start_time)

    # WEIGHT CONDITIONING
    start_time = time.time()
    try:
        metrics["weight_cond"] = compute_weight_conditioning(model)
    except Exception as e:
        print(f"[Conditioning] Error: {e}")
        metrics["weight_cond"] = 1.0
    timing["weight_cond"] = timing.get("weight_cond", 0) + (time.time() - start_time)

    # FISHER INFORMATION
    start_time = time.time()
    try:
        metrics["fisher"] = compute_fisher(model, inputs, targets)
    except Exception as e:
        print(f"[Fisher] Error: {e}")
        metrics["fisher"] = 0.0
    timing["fisher"] = timing.get("fisher", 0) + (time.time() - start_time)

    # GRASP
    start_time = time.time()
    try:
        metrics["grasp"] = compute_grasp_robust(model, inputs, targets)
    except Exception as e:
        print(f"[GraSP] Error: {e}")
        metrics["grasp"] = 0.0
    timing["grasp"] = timing.get("grasp", 0) + (time.time() - start_time)

    # SNIP
    start_time = time.time()
    try:
        metrics["snip"] = compute_snip(model, inputs, targets)
    except Exception as e:
        print(f"[SNIP] Error: {e}")
        metrics["snip"] = 0.0
    timing["snip"] = timing.get("snip", 0) + (time.time() - start_time)

    # JACOB COV
    start_time = time.time()
    try:
        metrics["jacob_cov"] = compute_jacob_cov(model, inputs)
    except Exception as e:
        print(f"[Jacob_Cov] Error: {e}")
        metrics["jacob_cov"] = 0.0
    timing["jacob_cov"] = timing.get("jacob_cov", 0) + (time.time() - start_time)

    # FLOPs count (computational complexity)
    start_time = time.time()
    try:
        metrics["flops"] = estimate_flops(model, inputs)
    except Exception as e:
        print(f"[FLOPs] Error: {e}")
        metrics["flops"] = 0.0
    timing["flops"] = timing.get("flops", 0) + (time.time() - start_time)

    return metrics


def aggregate_metrics(metrics, custom_weights=None, dataset_name=None):
    """
    Aggregate metrics with learned or predefined weights based on dataset.

    Args:
        metrics: Dict of computed metrics
        custom_weights: Optional custom weights
        dataset_name: Name of dataset to use preset weights
    """
    # Dataset-specific weights based on prior meta-analysis
    dataset_weights = {
        "cifar10": {
            "synflow": 0.35,
            "param_count": -0.05,
            "naswot": 0.25,
            "sensitivity": 0.2,
            "weight_cond": -0.15,
            "fisher": 0.3,
            "grasp": 0.2,
            "snip": 0.25,
            "jacob_cov": 0.15,
            "flops": -0.1,
        },
        "imagenet": {
            "synflow": 0.3,
            "param_count": -0.15,
            "naswot": 0.2,
            "sensitivity": 0.15,
            "weight_cond": -0.1,
            "fisher": 0.25,
            "grasp": 0.25,
            "snip": 0.2,
            "jacob_cov": 0.1,
            "flops": -0.15,
        },
        # Default weights for custom datasets
        "default": {
            "synflow": 0.3,
            "param_count": -0.1,
            "naswot": 0.3,
            "sensitivity": 0.2,
            "weight_cond": -0.1,
            "fisher": 0.25,
            "grasp": 0.2,
            "snip": 0.2,
            "jacob_cov": 0.1,
            "flops": -0.1,
        },
    }

    # Use provided weights, dataset-specific weights, or defaults
    if custom_weights is not None:
        active_weights = custom_weights
    elif dataset_name is not None and dataset_name in dataset_weights:
        active_weights = dataset_weights[dataset_name]
    else:
        active_weights = dataset_weights["default"]

    # Print the weights being used
    print("\nMetric weights:")
    for k, v in active_weights.items():
        if k in metrics:
            print(f"  {k}: {v}")

    # Normalize all metrics to comparable scale
    norm_metrics = {}
    for k, v in metrics.items():
        if k not in active_weights:
            continue

        if v != 0:
            # Apply appropriate normalization based on metric distribution
            if k in ["synflow", "param_count", "flops"]:
                # Log normalization for metrics with large ranges
                norm_metrics[k] = float(torch.log1p(torch.tensor(abs(v))).item())
            elif k in ["weight_cond"]:
                # Bounded metrics
                norm_metrics[k] = min(v, 100) / 100  # Cap at reasonable value
            else:
                # Other metrics
                norm_metrics[k] = float(v)
        else:
            norm_metrics[k] = 0.0

    # Calculate aggregate score from normalized metrics
    available_metrics = set(norm_metrics.keys()) & set(active_weights.keys())
    if not available_metrics:
        return 0.0

    agg_score = sum(norm_metrics[k] * active_weights[k] for k in available_metrics)
    return float(agg_score)


def detect_correlations(metrics_history, threshold=0.85):
    """
    Detect highly correlated metrics to avoid redundancy.

    Args:
        metrics_history: Dictionary with lists of metric values over multiple evaluations
        threshold: Correlation threshold above which metrics are considered redundant

    Returns:
        List of pairs of correlated metrics (to potentially downweight one)
    """
    # Calculate correlation matrix
    metric_names = list(metrics_history.keys())
    correlated_pairs = []

    # Need at least 2 data points for correlation
    if len(next(iter(metrics_history.values()))) < 2:
        return []

    # Convert to numpy arrays
    arrays = {k: np.array(v) for k, v in metrics_history.items()}

    # Calculate correlations
    for i, m1 in enumerate(metric_names):
        for j, m2 in enumerate(metric_names):
            if i < j:  # Check each pair only once
                if len(arrays[m1]) > 1 and len(arrays[m2]) > 1:
                    # Handle constant values which would cause division by zero
                    if np.std(arrays[m1]) == 0 or np.std(arrays[m2]) == 0:
                        continue

                    corr = np.corrcoef(arrays[m1], arrays[m2])[0, 1]
                    if abs(corr) > threshold:
                        correlated_pairs.append((m1, m2))

    return correlated_pairs


# === INDIVIDUAL METRIC IMPLEMENTATIONS ===


def compute_synflow(model, inputs):
    """
    Compute SynFlow score (Tanaka et al., 2020)
    Synaptic Flow: Network pruning that maximizes sensitivity flow
    """
    model.train()
    # Save original state
    original_state = {n: p.data.clone() for n, p in model.named_parameters()}

    # Set all parameters to their absolute values
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.abs()

    # Forward pass with dummy input
    dummy_input = torch.ones_like(inputs, requires_grad=True)
    output = model(dummy_input).sum()

    # Compute gradients
    output.backward()

    # Calculate score as sum of parameter-gradient products
    score = sum(
        (p * p.grad).sum().item()
        for p in model.parameters()
        if p.grad is not None and p.requires_grad
    )

    # Restore original parameters
    for name, p in model.named_parameters():
        if name in original_state:
            p.data = original_state[name]

    # Zero all gradients
    model.zero_grad()

    return score


def count_params(model):
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_naswot(model, inputs):
    """
    Compute NASWOT score (Mellor et al., 2021)
    Neural Architecture Search without Training
    """
    model.eval()
    activations = {}

    # Register forward hooks to capture activations
    def hook(m, inp, out):
        if isinstance(out, tuple):
            activations[m] = out[0]
        else:
            activations[m] = out

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))

    # Forward pass without gradient
    with torch.no_grad():
        _ = model(inputs)

    # Calculate NASWOT score
    score = 0
    for layer, act in activations.items():
        # Reshape activations
        act = act.reshape(act.size(0), -1)

        # Get binary activations (0/1 based on activation value)
        binary_act = (act > 0).float()

        # Calculate K matrix
        K = binary_act @ binary_act.t()

        # Add matrix rank to score
        try:
            score += torch.linalg.matrix_rank(K).item()
        except:
            # Fallback if SVD fails
            try:
                score += torch.matrix_rank(K).item()
            except:
                pass

    # Remove hooks
    for h in hooks:
        h.remove()

    return float(score)


def compute_sensitivity(model, inputs):
    """
    Compute sensitivity score: how sensitive the model is to input perturbations.
    Higher sensitivity might indicate better feature extraction capability.
    """
    model.train()
    model.zero_grad()

    # Forward pass
    inputs.requires_grad_(True)
    output = model(inputs)

    # Random target for gradient computation
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    loss.backward()

    # Compute input gradient magnitude
    input_grad = inputs.grad
    if input_grad is not None:
        input_grad_norm = (
            input_grad.norm(p=2, dim=tuple(range(1, input_grad.dim()))).mean().item()
        )
    else:
        input_grad_norm = 0.0

    # Compute parameter gradient magnitude
    param_grad_sum = sum(
        p.grad.abs().sum().item()
        for p in model.parameters()
        if p.grad is not None and p.requires_grad
    )
    param_count = sum(
        p.numel() for p in model.parameters() if p.grad is not None and p.requires_grad
    )

    # Combine input and parameter sensitivity
    param_sensitivity = param_grad_sum / max(param_count, 1)

    # Reset gradients and input requires_grad
    model.zero_grad()
    inputs.requires_grad_(False)

    return float(param_sensitivity + input_grad_norm)


def compute_weight_conditioning(model):
    """
    Compute average condition number of weight matrices.
    High condition numbers may indicate poor optimization landscape.
    """
    conds = []
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() >= 2 and p.requires_grad:
            # Reshape to 2D matrix
            W = p.view(p.size(0), -1)

            # Skip if matrix too small
            if min(W.size(0), W.size(1)) <= 1:
                continue

            try:
                # Compute SVD
                U, S, V = torch.linalg.svd(W, full_matrices=False)

                # Calculate condition number (ratio of largest/smallest singular value)
                if S[0] > 0 and S[-1] > 0:
                    cond = (S[0] / S[-1]).item()
                    conds.append(cond)
            except:
                continue

    if not conds:
        return 1.0  # Default if no condition numbers could be computed

    return float(sum(conds) / len(conds))


def compute_snip(model, inputs, targets):
    """
    SNIP metric (Lee et al., 2018)
    Single-shot Network Pruning based on connection sensitivity
    """
    criterion = nn.CrossEntropyLoss() if targets.dim() == 1 else nn.MSELoss()
    model.train()
    model.zero_grad()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # Calculate connection sensitivity
    snip_score = 0
    for name, p in model.named_parameters():
        if p.grad is not None and "weight" in name:
            snip_score += (p.grad * p).abs().sum().item()

    model.zero_grad()
    return float(snip_score)


def estimate_flops(model, inputs):
    """
    Estimate computational complexity (FLOPs) for the model.

    Note: This is a simplified FLOP counter that works for common layer types.
    For more accurate counts, consider using libraries like fvcore or ptflops.
    """

    # Define FLOP counters for different layer types
    def conv_flops(layer, input_shape, output_shape):
        batch_size = input_shape[0]
        in_channels = input_shape[1] if len(input_shape) > 1 else 1
        out_channels = output_shape[1] if len(output_shape) > 1 else output_shape[0]

        # Extract kernel size and other parameters based on layer dimension
        if isinstance(layer, nn.Conv1d):
            kernel_size = layer.kernel_size[0]
            output_size = output_shape[2]
        elif isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
            output_size = output_shape[2] * output_shape[3]
        elif isinstance(layer, nn.Conv3d):
            kernel_size = (
                layer.kernel_size[0] * layer.kernel_size[1] * layer.kernel_size[2]
            )
            output_size = output_shape[2] * output_shape[3] * output_shape[4]
        else:
            return 0

        # Compute MACs
        mac_count = (
            batch_size
            * out_channels
            * output_size
            * (in_channels // layer.groups * kernel_size)
        )
        # FLOPs = 2 * MACs (multiply-add)
        return mac_count * 2

    def linear_flops(layer, input_shape, output_shape):
        batch_size = input_shape[0]
        in_features = layer.in_features
        out_features = layer.out_features

        # Compute MACs
        mac_count = batch_size * in_features * out_features
        # FLOPs = 2 * MACs (multiply-add)
        return mac_count * 2

    # Track total FLOPs
    total_flops = 0

    # Register hooks to capture input/output shapes
    shapes = {}
    flops_per_module = {}

    def save_input_output_shape(name):
        def hook(module, input, output):
            input_shape = input[0].shape
            output_shape = (
                output.shape if not isinstance(output, tuple) else output[0].shape
            )
            shapes[name] = (input_shape, output_shape)

            # Calculate FLOPs based on module type
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                flops = conv_flops(module, input_shape, output_shape)
            elif isinstance(module, nn.Linear):
                flops = linear_flops(module, input_shape, output_shape)
            else:
                flops = 0

            flops_per_module[name] = flops

        return hook

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            hooks.append(module.register_forward_hook(save_input_output_shape(name)))

    # Forward pass to calculate shapes
    with torch.no_grad():
        _ = model(inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Sum FLOPs
    total_flops = sum(flops_per_module.values())

    return total_flops


# Function to track performance across architectures
def track_nas_performance(metrics_history, arch_name, metrics):
    """
    Track performance metrics across different architectures.
    Useful for meta-learning optimal weights.

    Args:
        metrics_history: Dictionary of historical metrics by architecture
        arch_name: Name of the current architecture
        metrics: Metrics for the current architecture
    """
    if arch_name not in metrics_history:
        metrics_history[arch_name] = {}

    # Store metrics
    for k, v in metrics.items():
        if k not in metrics_history[arch_name]:
            metrics_history[arch_name][k] = []
        metrics_history[arch_name][k].append(v)

    return metrics_history


# Function to adapt metric weights based on correlation with actual performance
def adapt_metric_weights(metrics_history, val_accuracies, epochs=10, lr=0.01):
    """
    Adapt metric weights based on correlation with validation performance.

    Args:
        metrics_history: Dictionary of metrics by architecture
        val_accuracies: Dictionary of validation accuracies by architecture
        epochs: Number of optimization epochs
        lr: Learning rate for weight adaptation

    Returns:
        Dictionary of optimized weights
    """
    # Extract features and target
    arch_names = list(val_accuracies.keys())

    if not arch_names:
        print("No architecture data available for weight adaptation")
        return None

    # Prepare feature matrix
    metric_names = set()
    for arch in arch_names:
        if arch in metrics_history:
            metric_names.update(metrics_history[arch].keys())

    metric_names = list(metric_names)

    # Filter out architectures with missing metrics
    valid_archs = []
    for arch in arch_names:
        if arch in metrics_history and all(
            m in metrics_history[arch] for m in metric_names
        ):
            valid_archs.append(arch)

    if not valid_archs:
        print("No architectures with complete metrics found")
        return None

    # Build feature matrix and target vector
    X = []
    y = []

    for arch in valid_archs:
        # Get metrics for this architecture
        arch_metrics = []
        for metric in metric_names:
            value = np.mean(metrics_history[arch][metric])
            arch_metrics.append(value)

        X.append(arch_metrics)
        y.append(val_accuracies[arch])

    X = np.array(X)
    y = np.array(y)

    # Normalize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Initialize weights
    weights = np.zeros(len(metric_names))

    # Simple gradient descent to optimize weights
    for epoch in range(epochs):
        # Predictions
        pred = X_norm @ weights

        # Loss (MSE)
        loss = np.mean((pred - y) ** 2)

        # Gradient
        grad = 2 * X_norm.T @ (pred - y) / len(y)

        # Update weights
        weights -= lr * grad

        if epoch % (epochs // 5) == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Convert to dictionary
    optimal_weights = {}
    for i, metric in enumerate(metric_names):
        optimal_weights[metric] = float(weights[i])

    # Normalize weights to reasonable magnitude
    total = sum(abs(w) for w in optimal_weights.values())
    if total > 0:
        for k in optimal_weights:
            optimal_weights[k] /= total

    return optimal_weights


########################################################
# end zero cost
##########################################################


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbels) / tau
    y_soft = F.softmax(y, dim=dim)

    if hard:
        index = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft


def train_darts_model(
    model,
    train_loader,
    val_loader,
    epochs=50,
    arch_learning_rate=3e-4,
    model_learning_rate=1e-3,
    arch_weight_decay=1e-3,
    model_weight_decay=1e-4,
    patience=10,
    tau_max=1.0,
    tau_min=0.1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    enable_logging=False,
    loss_type="huber",
    use_swa=False,
):
    """
    Enhanced training process for DARTS with fixed device handling for SWA
    """
    # Ensure model is on the correct device
    model = model.to(device)

    # Collect architecture parameters
    arch_params = []
    for cell in model.cells:
        for edge in cell.edges:
            arch_params.append(edge.alphas)

    # Optimizers
    arch_optimizer = torch.optim.Adam(
        arch_params,
        lr=arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=arch_weight_decay,
    )

    # Collect model parameters (excluding architecture parameters)
    model_params = []
    for name, param in model.named_parameters():
        if "alphas" not in name:
            model_params.append(param)

    model_optimizer = torch.optim.AdamW(
        model_params, lr=model_learning_rate, weight_decay=model_weight_decay
    )

    # Stochastic Weight Averaging - explicitly set device
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        # Ensure SWA model is on the same device
        swa_model = swa_model.to(device)
        swa_start = epochs // 2

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        model_optimizer, T_max=epochs
    )

    # Mixed precision
    scaler = GradScaler()

    # Training state
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    alpha_values = []

    # Temperature annealing for Gumbel-Softmax
    tau_schedule = np.linspace(tau_max, tau_min, epochs)

    # Loss function
    def calculate_loss(preds, targets):
        if loss_type == "huber":
            return F.huber_loss(preds, targets, delta=0.1)
        elif loss_type == "mse":
            return F.mse_loss(preds, targets)
        elif loss_type == "mae":
            return F.l1_loss(preds, targets)
        else:
            return F.mse_loss(preds, targets)  # Default to MSE

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Current temperature for Gumbel-Softmax
        tau = tau_schedule[epoch]

        # Track alpha values
        current_alphas = []
        for cell_idx, cell in enumerate(model.cells):
            for edge_idx, edge in enumerate(cell.edges):
                # Use current temperature
                weights = F.softmax(edge.alphas, dim=-1).detach().cpu().numpy()
                current_alphas.append((cell_idx, edge_idx, weights))
        alpha_values.append(current_alphas)

        # 1. Update architecture parameters (alphas)
        if epoch >= 1:  # Skip first epoch
            try:
                val_batch = next(iter(val_loader))
                val_x, val_y = val_batch[0].to(device), val_batch[1].to(device)

                arch_optimizer.zero_grad()

                # Forward pass - handle potential mixed precision issues
                val_x = val_x.float()  # Ensure float32 for arch update
                val_preds = model(val_x)
                val_loss = calculate_loss(val_preds, val_y)

                # Entropy regularization for arch params
                entropy_loss = 0.0
                for cell in model.cells:
                    for edge in cell.edges:
                        probs = F.softmax(edge.alphas, dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                        entropy_loss += entropy

                total_val_loss = val_loss - 0.01 * entropy_loss

                # Standard backprop without scaler for architecture update
                total_val_loss.backward()
                arch_optimizer.step()

            except StopIteration:
                print("Validation loader exhausted during architecture update")

        # 2. Update model weights
        for batch_idx, (batch_x, batch_y, *_) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            model_optimizer.zero_grad()

            # Handle mixed precision carefully
            with autocast(enabled=True):  # Explicitly enable
                # Force input to float32 for consistent operation with any ops
                # that might not support half precision
                batch_x_float = batch_x.float()
                preds = model(batch_x_float)
                loss = calculate_loss(preds, batch_y)

            # Scale loss and backprop
            scaler.scale(loss).backward()

            # Gradient clipping before optimizer step
            scaler.unscale_(model_optimizer)
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=5.0)

            scaler.step(model_optimizer)
            scaler.update()

            train_loss += loss.item()

            # Print progress
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        # Update learning rate
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Ensure float32 for validation
                batch_x_float = batch_x.float()
                preds = model(batch_x_float)
                loss = calculate_loss(preds, batch_y)

                val_loss += loss.item()

        # Calculate average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print epoch summary
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Check for improvement and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Store best model on CPU to avoid device mismatch later
            best_model_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }

            # Update SWA model if applicable
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Finalize SWA model if used
    if use_swa and epoch >= swa_start:
        print("Updating batch normalization statistics for SWA model...")

        # Create a custom loader just for updating batch norm on device
        def fix_device_dataloader():
            for batch_x, batch_y, *_ in train_loader:
                # Ensure batch is on the correct device and type
                yield batch_x.to(device).float(), batch_y.to(device)

        # Custom function to update batch norm statistics
        def update_bn_stats(loader, model, device=device):
            model.train()
            with torch.no_grad():
                for batch_x, _ in loader:
                    # Ensure all tensors are on the same device
                    if batch_x.device != device:
                        batch_x = batch_x.to(device)

                    # Forward pass to update BN statistics
                    model(batch_x)

        # Update batch norm statistics with custom function
        try:
            # First try PyTorch's implementation with our fixed dataloader
            torch.optim.swa_utils.update_bn(
                fix_device_dataloader(), swa_model, device=device
            )
        except Exception as e:
            print(f"Error using standard update_bn: {e}")
            print("Using custom batch norm update function...")
            # Fall back to our custom implementation
            update_bn_stats(fix_device_dataloader(), swa_model, device)

        # Evaluate SWA model
        swa_model.eval()
        swa_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_x_float = batch_x.float()
                swa_preds = swa_model(batch_x_float)
                loss = calculate_loss(swa_preds, batch_y)
                swa_val_loss += loss.item()

        swa_val_loss /= len(val_loader)
        print(
            f"Final SWA val loss: {swa_val_loss:.4f} vs Best non-SWA: {best_val_loss:.4f}"
        )

        if swa_val_loss < best_val_loss:
            print("SWA model performs better, using it as final model")
            best_model_state = {
                k: v.detach().clone() for k, v in swa_model.state_dict().items()
            }
            best_val_loss = swa_val_loss

    # Load best model (ensure it's on the right device)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)  # Ensure model is on the right device

    # Final model evaluation
    # print mse, rmse, mae, r2
    # Calculate metrics
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y, *_ in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_x_float = batch_x.float()
            preds = model(batch_x_float)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    y_pred_flat = all_preds.reshape(-1)
    y_true_flat = all_targets.reshape(-1)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    print(f"Final Model Performance:")
    print(f"Test Loss: {best_val_loss:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "alpha_values": alpha_values,
        "best_val_loss": best_val_loss,
    }


def expand_operations(
    initial_ops,
    op_importance,
    op_categories=None,
    complementary_map=None,
    top_k=3,
    max_ops=8,
    verbose=True,
):
    """
    Intelligently expand the set of architecture operations based on what performed well.

    Args:
        initial_ops (List[str]): Initial operation list.
        op_importance (List[Tuple[str, float]]): Ranked (op_name, score) list.
        op_categories (Dict[str, List[str]], optional): Category to ops map.
        complementary_map (Dict[str, List[str]], optional): Best-op to complementary ops.
        top_k (int): How many top operations to consider.
        max_ops (int): Maximum allowed operation count after expansion.
        verbose (bool): Whether to print reasoning.

    Returns:
        List[str]: Expanded operation list.
    """
    expanded_ops = list(initial_ops)
    top_ops = [op for op, _ in op_importance[:top_k]]
    best_op = top_ops[0]

    # 🧱 Default categories (no RNNs)
    if op_categories is None:
        op_categories = {
            "attention": ["Attention", "Transformer"],
            "spectral": ["Wavelet", "Fourier"],
            "convolution": ["TCN", "ConvMixer", "TimeConv"],
            "mlp": ["ResidualMLP", "Identity", "GRN"],
        }

    # 🤝 Complementary pairs (customized)
    if complementary_map is None:
        complementary_map = {
            "Attention": ["Wavelet", "GRN"],
            "Transformer": ["Fourier", "GRN"],
            "Wavelet": ["TimeConv", "ResidualMLP"],
            "Fourier": ["Wavelet", "GRN"],
            "TCN": ["GRN", "ResidualMLP"],
            "ConvMixer": ["GRN", "Wavelet"],
            "ResidualMLP": ["Attention", "TimeConv"],
            "Identity": ["ResidualMLP", "TimeConv"],
            "GRN": ["Attention", "ConvMixer"],
            "TimeConv": ["Wavelet", "ResidualMLP"],
        }

    def add_op(op, reason):
        if op not in expanded_ops:
            expanded_ops.append(op)
            if verbose:
                print(f"➕ Added {op} because {reason}")

    # Rule 1: Add all ops from any top-op's category
    for cat, ops in op_categories.items():
        if any(op in top_ops for op in ops):
            for op in ops:
                add_op(op, f"a {cat} op performed well")

    # Rule 2: Add complementary ops to the best op
    if best_op in complementary_map:
        for op in complementary_map[best_op]:
            add_op(op, f"it's complementary to {best_op}")

    # Rule 3: Ensure at least one op per category
    for cat, ops in op_categories.items():
        if not any(op in expanded_ops for op in ops):
            add_op(ops[0], f"ensuring {cat} coverage")

    # Rule 4: Limit total ops to max_ops
    if len(expanded_ops) > max_ops:

        def sort_key(op):
            if op in top_ops:
                return (0, top_ops.index(op))
            elif op in initial_ops:
                return (1, initial_ops.index(op))
            else:
                return (2, op)

        expanded_ops = sorted(expanded_ops, key=sort_key)[:max_ops]
        if verbose:
            print(f"⚠️ Trimmed to top {max_ops} ops for efficiency")

    return expanded_ops


def derive_final_architecture(model):
    """
    Create optimized model with fixed operations based on search results

    Args:
        model: Trained DARTS model

    Returns:
        Optimized model with fixed architecture
    """
    new_model = copy.deepcopy(model)

    # Replace mixed operations with fixed ones
    for cell in new_model.cells:
        new_edges = nn.ModuleList()
        for edge in cell.edges:
            weights = F.softmax(edge.alphas, dim=-1)
            top_op_idx = weights.argmax().item()
            top_op = edge.ops[top_op_idx]
            fixed_edge = FixedOp(top_op)
            new_edges.append(fixed_edge)
        cell.edges = new_edges

    return new_model


def train_final_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    device="cuda",
):
    """
    Train the final derived model with longer schedule and SWA

    Args:
        model: Derived model with fixed architecture
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to run on

    Returns:
        Dictionary with trained model and metrics
    """
    model = model.to(device)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # OneCycle learning rate for super-convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000,
    )

    # Stochastic Weight Averaging
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_model.to(device)
    swa_start = epochs // 3

    # Mixed precision
    scaler = GradScaler()

    # Training tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    print(f"Training final model for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_x, batch_y, *_ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            with autocast():
                preds = model(batch_x)
                loss = F.huber_loss(preds, batch_y, delta=0.1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                with autocast():
                    preds = model(batch_x)
                    loss = F.huber_loss(preds, batch_y, delta=0.1)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update SWA model
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                k: v.cpu().detach() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1
            if patience_counter >= 15:  # Higher patience for final model
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Apply SWA if applicable
    # Finalize SWA model if used
    if epoch >= swa_start:
        print("Updating batch normalization statistics for SWA model...")

        # Create a custom loader just for updating batch norm on device
        def fix_device_dataloader():
            for batch_x, batch_y, *_ in train_loader:
                # Ensure batch is on the correct device and type
                yield batch_x.to(device).float(), batch_y.to(device)

        # Custom function to update batch norm statistics
        def update_bn_stats(loader, model, device=device):
            model.train()
            with torch.no_grad():
                for batch_x, _ in loader:
                    # Ensure all tensors are on the same device
                    if batch_x.device != device:
                        batch_x = batch_x.to(device)

                    # Forward pass to update BN statistics
                    model(batch_x)

        # Update batch norm statistics with custom function
        try:
            # First try PyTorch's implementation with our fixed dataloader
            torch.optim.swa_utils.update_bn(
                fix_device_dataloader(), swa_model, device=device
            )
        except Exception as e:
            print(f"Error using standard update_bn: {e}")
            print("Using custom batch norm update function...")
            # Fall back to our custom implementation
            update_bn_stats(fix_device_dataloader(), swa_model, device)

        # Evaluate SWA model
        swa_model.eval()
        swa_val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                swa_model = swa_model.to(device)
                with autocast():
                    swa_preds = swa_model(batch_x).to(device)
                    loss = F.huber_loss(swa_preds, batch_y, delta=0.1)
                swa_val_loss += loss.item()

        swa_val_loss /= len(val_loader)
        print(
            f"SWA model validation loss: {swa_val_loss:.4f} vs Best: {best_val_loss:.4f}"
        )

        # Use SWA model if better
        if swa_val_loss < best_val_loss:
            print("SWA model performs better! Using SWA model.")
            best_model_state = {
                k: v.cpu().detach() for k, v in swa_model.state_dict().items()
            }
            best_val_loss = swa_val_loss

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, *_ in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with autocast():
                preds = model(batch_x)
                loss = F.huber_loss(preds, batch_y, delta=0.1)

            test_loss += loss.item()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader)

    training_time = time.time() - start_time
    print(f"Model training completed in {training_time:.1f} seconds")
    print(f"Test loss: {test_loss:.4f}")

    # print mse, rmse, mae, mape, r2
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    mape = (
        np.mean(np.abs((all_preds - all_targets) / (np.abs(all_targets) + 1e-5))) * 100
    )
    print(f"Final Model Performance:")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": test_loss,
        "training_time": training_time,
    }


def multi_fidelity_darts_search(
    input_dim=3,
    hidden_dims=[32, 64, 128],  # Multiple model sizes
    forecast_horizon=6,
    seq_length=12,
    device="cuda" if torch.cuda.is_available() else "cpu",
    train_loader=None,
    val_loader=None,
    test_loader=None,
    num_candidates=10,  # Number of architectures to evaluate
    full_train_epochs=30,  # Epochs for final training
):
    """
    Multi-fidelity architecture search using zero-cost metrics
    for efficient screening
    """
    print("Starting multi-fidelity DARTS search...")

    # Define possible operations
    all_ops = [
        "Identity",
        "TimeConv",
        "GRN",
        "Wavelet",
        "Fourier",
        "Attention",
        "TCN",
        "ResidualMLP",
        "ConvMixer",
    ]

    # Phase 1: Generate random architectures and evaluate with zero-cost metrics
    candidates = []

    for i in range(num_candidates):
        # Randomly select operations
        num_ops = random.randint(5, len(all_ops))
        selected_ops = ["Identity"]  # Always include Identity
        selected_ops.extend(
            random.sample([op for op in all_ops if op != "Identity"], num_ops - 1)
        )

        # Randomly select model size
        hidden_dim = random.choice(hidden_dims)

        # Randomly select num_cells and num_nodes
        num_cells = random.randint(1, 2)
        num_nodes = random.randint(2, 4)

        # Create model
        model = TimeSeriesDARTS(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=hidden_dim,
            forecast_horizon=forecast_horizon,
            seq_length=seq_length,
            num_cells=num_cells,
            num_nodes=num_nodes,
            selected_ops=selected_ops,
        ).to(device)

        # Evaluate with zero-cost metrics
        print(f"Evaluating candidate {i+1}/{num_candidates}...")
        metrics = evaluate_zero_cost_metrics(model, val_loader, device)

        candidates.append(
            {
                "model": model,
                "metrics": metrics,
                "score": metrics["aggregate_score"],
                "selected_ops": selected_ops,
                "hidden_dim": hidden_dim,
                "num_cells": num_cells,
                "num_nodes": num_nodes,
            }
        )

    # Phase 2: Select top candidates based on zero-cost metrics
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_k = min(5, len(candidates))
    top_candidates = candidates[:top_k]

    print(f"\nSelected top {top_k} candidates based on zero-cost metrics:")
    for i, candidate in enumerate(top_candidates):
        print(f"Candidate {i+1}:")
        print(f"  Score: {candidate['score']:.4f}")
        print(f"  Operations: {candidate['selected_ops']}")
        print(f"  Hidden dim: {candidate['hidden_dim']}")
        print(
            f"  Architecture: {candidate['num_cells']} cells, {candidate['num_nodes']} nodes"
        )

    # Phase 3: Train top candidates with DARTS for a short period
    trained_candidates = []
    for i, candidate in enumerate(top_candidates):
        print(f"\nShort DARTS training for candidate {i+1}/{top_k}...")

        # Quick DARTS training
        results = train_darts_model(
            model=candidate["model"],
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=full_train_epochs // 3,  # Short training
            device=device,
            use_swa=False,
        )

        # Derive architecture
        derived_model = derive_final_architecture(results["model"]).to(device)

        # Evaluate on validation set
        val_loss = 0.0
        derived_model.eval()
        with torch.no_grad():
            for batch_x, batch_y, *_ in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = derived_model(batch_x)
                loss = F.huber_loss(preds, batch_y, delta=0.1)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        trained_candidates.append(
            {"model": derived_model, "val_loss": val_loss, "candidate": candidate}
        )

    # Phase 4: Select best candidate and train fully
    trained_candidates.sort(key=lambda x: x["val_loss"])
    best_candidate = trained_candidates[0]

    print(f"\nBest candidate found with val_loss: {best_candidate['val_loss']:.6f}")
    print(f"Selected operations: {best_candidate['candidate']['selected_ops']}")

    # Create final derived architecture
    print(f"\nDeriving final architecture from search results")
    derived_model = best_candidate["model"].to(device)
    final_model = copy.deepcopy(derived_model)

    # train the final model with the derived architecture
    print(f"\nTraining derived model with final architecture")
    final_epochs = 100
    print(f"\nPHASE 3: Training Final Model ({final_epochs} epochs)")

    # print the final model architecture
    print(final_model)
    print(pato)

    # Enhanced training configuration
    advanced_config = {
        "optimizer": "Adam",
        "learning_rate": 5e-4,
        "weight_decay": 1e-5,
        "scheduler": "OneCycleLR",
        "mixed_precision": True,
        "gradient_clip": 1.0,
        "dropout": 0.2,
        "stochastic_weight_averaging": True,
        "batch_size": 256,
    }

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=advanced_config["learning_rate"],
        weight_decay=advanced_config["weight_decay"],
    )

    # Use OneCycleLR scheduler for super-convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=advanced_config["learning_rate"],
        epochs=final_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # Set up stochastic weight averaging
    if advanced_config["stochastic_weight_averaging"]:
        swa_model = torch.optim.swa_utils.AveragedModel(final_model)
        swa_start = final_epochs // 4

    # Move model to device
    final_model = final_model.to(device)

    # Set up gradient scaler for mixed precision
    scaler = GradScaler()

    # Training parameters
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    # Training loop with progress bar
    from tqdm import tqdm

    for epoch in range(final_epochs):
        # Training phase
        final_model.train()
        train_loss = 0.0
        train_batches = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{final_epochs} [Train]"
        )

        for batch_x, batch_y, *_ in train_batches:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Use mixed precision
            with autocast():
                # Forward pass
                preds = final_model(batch_x)

                # Calculate loss - multi-horizon weighted loss
                if batch_y.dim() > 2 and batch_y.size(1) > 1:
                    # Weighted loss with higher weights for near-term predictions
                    horizon_weights = torch.linspace(
                        1.0, 0.5, batch_y.size(1), device=device
                    )
                    losses = F.huber_loss(preds, batch_y, delta=0.1, reduction="none")
                    weighted_losses = losses * horizon_weights.view(1, -1, 1)
                    loss = weighted_losses.mean()
                else:
                    loss = F.huber_loss(preds, batch_y, delta=0.1)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping
            if advanced_config["gradient_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    final_model.parameters(), max_norm=advanced_config["gradient_clip"]
                )

            scaler.step(optimizer)
            scaler.update()

            # Step scheduler every batch (OneCycleLR)
            scheduler.step()

            train_loss += loss.item()
            train_batches.set_postfix({"loss": f"{loss.item():.4f}"})

        # Average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        final_model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        val_batches = tqdm(val_loader, desc=f"Epoch {epoch+1}/{final_epochs} [Valid]")

        with torch.no_grad():
            for batch_x, batch_y, *_ in val_batches:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                with autocast():
                    preds = final_model(batch_x, teacher_forcing_ratio=0.0)
                    loss = F.huber_loss(preds, batch_y, delta=0.1)

                val_loss += loss.item()
                val_batches.set_postfix({"loss": f"{loss.item():.4f}"})

                # Collect predictions for metrics
                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Update SWA model if applicable
        if advanced_config["stochastic_weight_averaging"] and epoch >= swa_start:
            swa_model.update_parameters(final_model)

        # Print progress
        print(
            f"Epoch {epoch+1}/{final_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                k: v.cpu().detach() for k, v in final_model.state_dict().items()
            }
            print(f"  → New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= 15:  # Increased patience
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state is not None:
        final_model.load_state_dict(best_model_state)

    # Plot training curve for the final model
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss")
    plt.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Final Model Training Progress", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("final_model_training.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Phase 4: Evaluate on test set with enhanced metrics
    print(f"\nPHASE 4: Evaluating Final Model on Test Set")
    final_model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y, *_ in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with autocast():
                preds = final_model(batch_x)

            loss = F.huber_loss(preds, batch_y, delta=0.1)
            test_loss += loss.item()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    test_loss /= len(test_loader)

    # Calculate comprehensive metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate overall metrics
    y_pred_flat = all_preds.reshape(-1)
    y_true_flat = all_targets.reshape(-1)

    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = (
        np.mean(np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-5)))
        * 100
    )
    r2 = r2_score(y_true_flat, y_pred_flat)

    # Print final results
    print(f"\nOverall Model Performance:")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.6f}")

    return {
        "final_model": final_model,
        "candidates": candidates,
        "top_candidates": top_candidates,
        "trained_candidates": trained_candidates,
        "best_candidate": best_candidate,
    }
