import contextlib
import re
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class Config:
    """Unified configuration"""
    max_samples: int = 32
    max_outputs: int = 10
    eps: float = 1e-8
    timeout: float = 30.0
    enable_mixed_precision: bool = False
    weights: Dict[str, float] = field(default_factory=lambda: {
        'synflow': 0.25, 'grasp': 0.20, 'fisher': 0.20, 'jacobian': 0.15,
        'naswot': 0.15, 'snip': 0.15, 'params': -0.05, 
        'conditioning': -0.10, 'flops': -0.05, 'sensitivity': 0.10,
        'zennas': 0.10
    })


@dataclass
class Result:
    """Metric computation result"""
    value: float
    success: bool = True
    error: str = ""
    time: float = 0.0
    
    def __repr__(self):
        status = "✓" if self.success else "✗"
        return f"Result({status} {self.value:.4f}, {self.time:.3f}s)"


class CompatibilityHelper:
    """Handles model compatibility issues"""
    
    @staticmethod
    @contextlib.contextmanager
    def safe_mode(model):
        """Context manager for safe metric computation"""
        # Disable problematic features
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        
        # Handle transformer attention if needed
        original_sdpa = None
        if hasattr(F, "scaled_dot_product_attention"):
            original_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = CompatibilityHelper._manual_attention
        
        try:
            yield
        finally:
            torch.backends.cudnn.enabled = prev_cudnn
            if original_sdpa:
                F.scaled_dot_product_attention = original_sdpa
    
    @staticmethod
    def _manual_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        """Manual attention implementation for compatibility"""
        if scale is None:
            scale = 1.0 / (q.size(-1) ** 0.5)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn = attn + attn_mask
        elif is_causal:
            seq_len = q.size(-2)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            attn = attn.masked_fill(mask.bool(), float("-inf"))
        
        attn = F.softmax(attn, dim=-1)
        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)
        
        return torch.matmul(attn, v)
    
    @staticmethod
    def prepare_data(outputs, targets):
        """Handle common output-target mismatches"""
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        
        # For sequence tasks, handle 3D tensors more carefully
        if outputs.dim() == 3 and targets.dim() == 3:
            # Both are sequences - check if shapes match
            if outputs.shape == targets.shape:
                # Shapes match, keep as is for sequence-to-sequence
                pass
            elif outputs.size(1) != targets.size(1):
                # Different sequence lengths - take last timestep for both
                outputs = outputs[:, -1]
                targets = targets[:, -1]
            else:
                # Same sequence length but different features - flatten
                outputs = outputs.view(outputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)
        elif outputs.dim() == 3 and targets.dim() <= 2:
            # Output is sequence, target is not - take last timestep of output
            outputs = outputs[:, -1]
        elif outputs.dim() <= 2 and targets.dim() == 3:
            # Target is sequence, output is not - take last timestep of target
            targets = targets[:, -1]
        
        # Handle classification vs regression
        if targets.dtype == torch.long:
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            if targets.dim() > 1 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            
            # Handle binary classification case
            if outputs.size(1) == 1 and targets.max() <= 1:
                outputs = outputs.squeeze(1)
        else:
            # For regression, ensure shapes match
            if outputs.shape != targets.shape:
                if outputs.dim() == 1 and targets.dim() == 2:
                    outputs = outputs.unsqueeze(1)
                elif outputs.dim() == 2 and targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                elif outputs.numel() == targets.numel():
                    # Same number of elements, reshape to match
                    target_shape = targets.shape
                    outputs = outputs.view(target_shape)
        
        return outputs, targets
    
    @staticmethod
    def get_loss_fn(targets):
        """Get appropriate loss function"""
        return nn.CrossEntropyLoss() if targets.dtype == torch.long else nn.MSELoss()


class MetricsComputer:
    """Computes all zero-cost NAS metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.helper = CompatibilityHelper()
    
    def _compute_safely(self, fn, *args, **kwargs) -> Result:
        """Safe computation wrapper with detailed error reporting"""
        start = time.time()
        try:
            value = fn(*args, **kwargs)
            return Result(float(value), True, "", time.time() - start)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return Result(0.0, False, error_msg, time.time() - start)
    
    def synflow(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """SynFlow metric"""
        def _compute():
            model.train()
            
            # Save and set absolute weights
            original = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    original[name] = param.data.clone()
                    param.data = param.data.abs()
            
            try:
                dummy = torch.ones_like(inputs, requires_grad=True)
                with self.helper.safe_mode(model):
                    output = model(dummy)
                    if isinstance(output, tuple):
                        output = output[0]
                    output.sum().backward()
                
                score = sum((p * p.grad).sum().item() 
                           for p in model.parameters() 
                           if p.requires_grad and p.grad is not None)
                return abs(score)
            finally:
                for name, param in model.named_parameters():
                    if name in original:
                        param.data = original[name]
                model.zero_grad()
        
        return self._compute_safely(_compute)
    
    def grasp(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> Result:
        """GraSP metric"""
        def _compute():
            was_training = model.training
            model.train()
            
            try:
                x, y = inputs.clone().detach(), targets.clone().detach()
                loss_fn = self.helper.get_loss_fn(y)
                
                # Clear any existing gradients
                model.zero_grad()
                
                with self.helper.safe_mode(model):
                    outputs = model(x)
                    outputs, y = self.helper.prepare_data(outputs, y)
                    
                    # Additional shape validation
                    if y.dtype == torch.long:
                        if outputs.dim() == 2 and y.dim() == 1:
                            if outputs.size(1) == 1:
                                # Binary classification case
                                outputs = outputs.squeeze(1)
                                loss_fn = nn.BCEWithLogitsLoss()
                            elif y.max() >= outputs.size(1):
                                raise ValueError(f"Target class {y.max().item()} >= num_classes {outputs.size(1)}")
                    
                    loss = loss_fn(outputs, y)
                    
                    # Get trainable parameters
                    params = [p for p in model.parameters() if p.requires_grad]
                    if not params:
                        raise ValueError("No trainable parameters found")
                    
                    # First-order gradients with graph retention
                    grads = torch.autograd.grad(
                        loss, params, create_graph=True, retain_graph=True, allow_unused=True
                    )
                    grads = [g for g in grads if g is not None]
                    
                    if not grads:
                        raise ValueError("No gradients computed in first-order pass")
                    
                    # Gradient magnitude and second-order gradients
                    grad_mag = sum(g.pow(2).sum() for g in grads)
                    
                    # Clear gradients before second-order computation
                    model.zero_grad()
                    grad_mag.backward(retain_graph=False)
                    
                    score = sum((p * p.grad).sum().item() 
                               for p in model.parameters() 
                               if p.requires_grad and p.grad is not None)
                    return abs(score)
                        
            finally:
                model.zero_grad()
                if not was_training:
                    model.eval()
        
        return self._compute_safely(_compute)
        
    def fisher(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> Result:
        """Fisher Information metric (fixed for both classification and regression)"""
        def _compute():
            was_training = model.training
            model.eval()
            
            try:
                x, y = inputs.clone().detach(), targets.clone().detach()
                loss_fn = self.helper.get_loss_fn(y)

                fisher_score = 0.0
                batch_size = min(x.size(0), self.config.max_samples)

                for i in range(batch_size):
                    model.zero_grad()

                    # Get single sample input/target
                    input_i = x[i:i+1]
                    target_i = y[i:i+1]

                    with self.helper.safe_mode(model):
                        outputs_i = model(input_i)
                        outputs_i, target_i = self.helper.prepare_data(outputs_i, target_i)

                        loss = loss_fn(outputs_i, target_i)
                        loss.backward()

                        for param in model.parameters():
                            if param.grad is not None:
                                fisher_score += param.grad.pow(2).sum().item()

                return fisher_score / batch_size

            finally:
                model.zero_grad()
                if not was_training:
                    model.eval()

        return self._compute_safely(_compute)

    def jacobian(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """Jacobian Covariance Entropy metric (with NaN detection and logging)"""
        def _compute():
            was_training = model.training
            model.train()

            try:
                batch_size = min(inputs.size(0), self.config.max_samples)
                x = inputs[:batch_size].detach().clone().requires_grad_(True)

                with self.helper.safe_mode(model):
                    outputs = model(x)
                    outputs, _ = self.helper.prepare_data(outputs, torch.zeros(batch_size))

                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)

                    output_size = min(outputs.size(1), self.config.max_outputs)
                    input_size = x.view(batch_size, -1).size(1)

                    jacobian = torch.zeros(batch_size, output_size, input_size, device=x.device)

                    for i in range(output_size):
                        model.zero_grad()
                        x.grad = None
                        outputs[:, i].sum().backward(retain_graph=(i < output_size - 1))
                        if x.grad is not None:
                            for j in range(batch_size):
                                jacobian[j, i] = x.grad[j].flatten()

                    score = 0.0
                    valid_samples = 0

                    for j in range(batch_size):
                        J = jacobian[j]
                        mask = J.abs().sum(dim=1) > self.config.eps

                        if mask.sum() > 1:
                            J_f = J[mask]
                            JJ = J_f @ J_f.t() + self.config.eps * torch.eye(J_f.size(0), device=J.device)

                            try:
                                eigenvals = torch.linalg.eigvalsh(JJ)
                                eigenvals = eigenvals[eigenvals > self.config.eps]

                                if len(eigenvals) > 0:
                                    eigenvals_norm = eigenvals / eigenvals.sum()
                                    entropy = -(eigenvals_norm * torch.log(eigenvals_norm + self.config.eps)).sum()

                                    if torch.isnan(entropy) or torch.isinf(entropy):
                                        print(f"[jacobian] Sample {j}: NaN or Inf entropy detected, skipping.")
                                        continue

                                    score += entropy.item()
                                    valid_samples += 1
                            except Exception as e:
                                print(f"[jacobian] Sample {j} eigenval computation failed: {e}")
                                continue

                    return score / max(valid_samples, 1)

            finally:
                x.requires_grad_(False)
                model.zero_grad()
                if not was_training:
                    model.eval()

        return self._compute_safely(_compute)

    def naswot(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """NASWOT metric (average across batches + spectral norm)"""
        def _compute():
            model.eval()
            activations = {}

            def hook(module, inp, out):
                activations[module] = out[0] if isinstance(out, tuple) else out

            hooks = []
            for module in model.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    hooks.append(module.register_forward_hook(hook))

            try:
                with torch.no_grad():
                    model(inputs)

                total_rank = 0.0
                spectral_sum = 0.0
                layer_count = 0

                for activation in activations.values():
                    try:
                        act = activation.reshape(activation.size(0), -1)
                        binary = (act > 0).float()

                        # Kernel matrix
                        K = binary @ binary.t()
                        K += self.config.eps * torch.eye(K.size(0), device=K.device)

                        # Rank
                        rank = torch.linalg.matrix_rank(K).item()
                        total_rank += rank

                        # Spectral norm (top eigenvalue)
                        eigvals = torch.linalg.eigvalsh(K)
                        spectral_norm = eigvals[-1].item() if eigvals.numel() > 0 else 0.0
                        spectral_sum += spectral_norm

                        layer_count += 1

                    except Exception as e:
                        print(f"[naswot] Layer failed with error: {e}")
                        continue

                if layer_count == 0:
                    raise ValueError("No valid layers found for NASWOT computation")

                # You can choose to return just rank, just spectral norm, or both
                return (total_rank + spectral_sum) / (2 * layer_count)

            finally:
                for hook in hooks:
                    hook.remove()

        return self._compute_safely(_compute)


    def snip(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> Result:
        """SNIP metric"""
        def _compute():
            was_training = model.training
            model.train()
            
            try:
                x, y = inputs.clone().detach(), targets.clone().detach()
                loss_fn = self.helper.get_loss_fn(y)
                
                # Clear gradients
                model.zero_grad()
                
                with self.helper.safe_mode(model):
                    outputs = model(x)
                    outputs, y = self.helper.prepare_data(outputs, y)
                    
                    # Additional shape validation
                    if y.dtype == torch.long:
                        if outputs.dim() == 2 and y.dim() == 1:
                            if outputs.size(1) == 1:
                                # Binary classification case
                                outputs = outputs.squeeze(1)
                                loss_fn = nn.BCEWithLogitsLoss()
                            elif y.max() >= outputs.size(1):
                                raise ValueError(f"Target class {y.max().item()} >= num_classes {outputs.size(1)}")
                    
                    loss = loss_fn(outputs, y)
                    loss.backward()
                
                # Calculate SNIP score with better error handling
                score = 0.0
                param_count = 0
                weight_layers_found = 0
                
                for name, param in model.named_parameters():
                    if param.requires_grad and 'weight' in name:
                        weight_layers_found += 1
                        if param.grad is not None:
                            param_score = (param.grad * param).abs().sum().item()
                            if not (torch.isnan(torch.tensor(param_score)) or torch.isinf(torch.tensor(param_score))):
                                score += param_score
                                param_count += 1
                
                if weight_layers_found == 0:
                    raise ValueError("No weight layers found in model")
                if param_count == 0:
                    raise ValueError("No valid gradients found for weight parameters")
                
                # Return normalized score if we have valid parameters
                return score / max(param_count, 1)
                
            finally:
                model.zero_grad()
                if not was_training:
                    model.eval()
        
        return self._compute_safely(_compute)
    
    def params(self, model: nn.Module) -> Result:
        """Parameter count"""
        def _compute():
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return self._compute_safely(_compute)
    
    def conditioning(self, model: nn.Module) -> Result:
        """Weight conditioning"""
        def _compute():
            conditions = []
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() >= 2 and param.requires_grad:
                    W = param.view(param.size(0), -1)
                    if min(W.size()) > 1:
                        try:
                            _, S, _ = torch.linalg.svd(W, full_matrices=False)
                            if S[0] > 0 and S[-1] > 0:
                                conditions.append((S[0] / S[-1]).item())
                        except:
                            continue
            return sum(conditions) / len(conditions) if conditions else 1.0
        
        return self._compute_safely(_compute)
    
    def flops(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """FLOP estimation"""
        def _compute():
            flops_count = {}
            
            def counting_hook(name):
                def hook(module, inp, out):
                    input_shape = inp[0].shape
                    output_shape = out.shape if not isinstance(out, tuple) else out[0].shape
                    
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        kernel_ops = np.prod(module.kernel_size) * module.in_channels // module.groups
                        output_elements = np.prod(output_shape)
                        flops = output_elements * kernel_ops * 2
                    elif isinstance(module, nn.Linear):
                        flops = input_shape[0] * module.in_features * module.out_features * 2
                    else:
                        flops = 0
                    
                    flops_count[name] = flops
                return hook
            
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    hooks.append(module.register_forward_hook(counting_hook(name)))
            
            try:
                with torch.no_grad():
                    model(inputs)
                return sum(flops_count.values())
            finally:
                for hook in hooks:
                    hook.remove()
        
        return self._compute_safely(_compute)
    
    def sensitivity(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """Input sensitivity"""
        def _compute():
            model.train()
            model.zero_grad()
            
            x = inputs.clone().detach().requires_grad_(True)
            output = model(x)
            target = torch.randn_like(output)
            loss = F.mse_loss(output, target)
            loss.backward()
            
            # Input gradient norm
            input_grad_norm = x.grad.norm(p=2, dim=tuple(range(1, x.grad.dim()))).mean().item() if x.grad is not None else 0.0
            
            # Parameter gradient norm
            param_grad_sum = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None and p.requires_grad)
            param_count = sum(p.numel() for p in model.parameters() if p.grad is not None and p.requires_grad)
            param_sensitivity = param_grad_sum / max(param_count, 1)
            
            model.zero_grad()
            x.requires_grad_(False)
            
            return param_sensitivity + input_grad_norm
        
        return self._compute_safely(_compute)
    
    def compute_all(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor = None) -> Dict[str, Result]:
        """Compute all metrics"""
        results = {}
        
        # Model-only metrics
        for metric in ['params', 'conditioning']:
            results[metric] = getattr(self, metric)(model)
        
        # Model + input metrics
        for metric in ['synflow', 'naswot', 'jacobian', 'flops', 'sensitivity', 'zennas']:
            results[metric] = getattr(self, metric)(model, inputs)
        
        # Target-dependent metrics
        if targets is not None:
            for metric in ['grasp', 'fisher', 'snip']:
                results[metric] = getattr(self, metric)(model, inputs, targets)
        
        return results

    def zennas(self, model: nn.Module, inputs: torch.Tensor) -> Result:
        """Zen-NAS ReLU activation signal-to-noise ratio"""
        def _compute():
            model.eval()
            activations = []

            def hook_fn(_, __, output):
                act = output.detach()
                activations.append(act.view(act.size(0), -1))

            hooks = []
            for module in model.modules():
                if isinstance(module, nn.ReLU):
                    hooks.append(module.register_forward_hook(hook_fn))

            try:
                with torch.no_grad():
                    model(inputs)

                score = 0.0
                count = 0
                for act in activations:
                    mean = act.mean(dim=1)
                    std = act.std(dim=1) + self.config.eps
                    snr = (mean ** 2 / std ** 2).mean().item()
                    if not torch.isnan(torch.tensor(snr)):
                        score += snr
                        count += 1

                return score / max(count, 1)

            finally:
                for hook in hooks:
                    hook.remove()

        return self._compute_safely(_compute)


class ZeroCostNAS:
    """Main zero-cost NAS evaluation class"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.computer = MetricsComputer(self.config)
    
    def evaluate_model(self, model: nn.Module, dataloader: DataLoader, device: torch.device, num_batches: int = 3) -> Dict[str, Any]:
        """Evaluate a single model"""
        # Get sample batches
        batches = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch if not isinstance(batch, (list, tuple)) else batch[0]
                # Create dummy targets based on model output shape
                with torch.no_grad():
                    model.eval()
                    dummy_output = model(inputs[:1].to(device))
                    if isinstance(dummy_output, tuple):
                        dummy_output = dummy_output[0]
                    
                    # Create appropriate dummy targets
                    if dummy_output.dim() > 1 and dummy_output.size(1) > 1:
                        # Multi-class classification
                        targets = torch.randint(0, dummy_output.size(1), (inputs.size(0),))
                    else:
                        # Binary classification or regression
                        targets = torch.randint(0, 2, (inputs.size(0),)) if dummy_output.dim() > 1 else torch.randn(inputs.size(0))
            
            inputs = inputs[:self.config.max_samples].to(device)
            targets = targets[:self.config.max_samples].to(device)
            batches.append((inputs, targets))
        
        # Compute metrics across batches
        all_results = []
        for i, (inputs, targets) in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}")
            batch_results = self.computer.compute_all(model, inputs, targets)
            all_results.append(batch_results)
        
        # Aggregate results
        final_results = self._aggregate_results(all_results)
        
        # Compute weighted score
        score = self._compute_score(final_results)
        
        return {
            'metrics': {name: result.value for name, result in final_results.items()},
            'success_rates': {name: result.success for name, result in final_results.items()},
            'error_messages': {name: result.error for name, result in final_results.items() if not result.success},
            'aggregate_score': score,
            'config': self.config
        }
    
    def compare(self, models: List[nn.Module], names: List[str], dataloader: DataLoader, device: torch.device) -> Dict[str, Any]:
        """Compare multiple models"""
        results = {}
        for model, name in zip(models, names):
            print(f"\nEvaluating {name}...")
            results[name] = self.evaluate_model(model, dataloader, device)
        
        # Create rankings
        rankings = {}
        for metric in results[names[0]]['metrics'].keys():
            values = [(name, results[name]['metrics'][metric]) for name in names]
            reverse = metric not in ['params', 'conditioning', 'flops']  # Higher is better except for these
            values.sort(key=lambda x: x[1], reverse=reverse)
            rankings[metric] = [name for name, _ in values]
        
        # Overall ranking
        scores = [(name, results[name]['aggregate_score']) for name in names]
        scores.sort(key=lambda x: x[1], reverse=True)
        rankings['overall'] = [name for name, _ in scores]
        
        return {
            'results': results,
            'rankings': rankings,
            'best': rankings['overall'][0],
            'summary': {name: {'score': results[name]['aggregate_score'], 'rank': rankings['overall'].index(name) + 1} for name in names}
        }
    
    def _aggregate_results(self, all_results: List[Dict[str, Result]]) -> Dict[str, Result]:
        """Aggregate results across batches using median"""
        aggregated = {}
        
        for metric in all_results[0].keys():
            values = [r[metric].value for r in all_results if r[metric].success]
            times = [r[metric].time for r in all_results]
            successes = [r[metric].success for r in all_results]
            
            if values:
                value = float(np.median(values))
                success = True
            else:
                value = 0.0
                success = False
            
            aggregated[metric] = Result(value, success, "", sum(times) / len(times))
        
        return aggregated
    
    def _compute_score(self, results: Dict[str, Result]) -> float:
        """Compute weighted aggregate score"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, result in results.items():
            if result.success and metric in self.config.weights:
                weight = self.config.weights[metric]
                
                # Normalize values
                if metric in ['synflow', 'params', 'flops']:
                    normalized = np.log1p(abs(result.value))
                elif metric == 'conditioning':
                    normalized = min(result.value, 100.0) / 100.0
                else:
                    normalized = result.value
                
                total_score += normalized * weight
                total_weight += abs(weight)
        
        return total_score / max(total_weight, 1.0)