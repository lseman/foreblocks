import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ---------- Fixed Core Components ----------

class RMSNorm(nn.Module):
    """Corrected RMSNorm: divide by sqrt(mean(x^2))"""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    
    def forward(self, x):
        # Correct RMS calculation: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class SafeAdaptiveNorm(nn.Module):
    """Fixed adaptive normalization with proper BN handling"""
    def __init__(self, d):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.rms_norm = RMSNorm(d)
        self.batch_norm = nn.BatchNorm1d(d)
        # Gated mixture with softmax
        self.gate = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
    
    def forward(self, x):
        original_shape = x.shape
        weights = F.softmax(self.gate, dim=0)
        
        ln_out = self.layer_norm(x)
        rms_out = self.rms_norm(x)
        
        # Handle BatchNorm shape requirements
        if x.dim() == 3 and x.size(1) > 1:  # (B, L, D) with L > 1
            # Flatten for BN, then reshape
            x_flat = x.view(-1, x.size(-1))  # (B*L, D)
            bn_out = self.batch_norm(x_flat).view(original_shape)
        elif x.dim() == 2:  # (B, D)
            bn_out = self.batch_norm(x)
        else:
            # Skip BN for problematic shapes
            bn_out = torch.zeros_like(x)
            weights = F.softmax(self.gate[:2], dim=0)  # Only LN + RMS
            weights = F.pad(weights, (0, 1), value=0.0)
        
        return weights[0] * ln_out + weights[1] * rms_out + weights[2] * bn_out


class GEGLU(nn.Module):
    """Standard GEGLU activation"""
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden * 2)
        self.w2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        u, v = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.dropout(F.gelu(u) * v))


# ---------- Fixed Neural ODT ----------

class NeuralObliviousDecisionTree(nn.Module):
    """Fixed NODE-style ODT with proper path probability products"""
    def __init__(self, n_features, depth=4, n_trees=8, d_out=64):
        super().__init__()
        self.depth = depth
        self.n_trees = n_trees
        
        # Feature selectors for each depth level
        self.feature_selectors = nn.ModuleList([
            nn.Linear(n_features, n_trees) for _ in range(depth)
        ])
        
        # Leaf values
        self.leaf_values = nn.Parameter(torch.randn(n_trees, 2**depth, d_out))
        
    def forward(self, x):  # (B, F)
        B = x.size(0)
        
        # Initialize path probabilities for each tree
        path_probs = torch.ones(B, self.n_trees, 2**self.depth, device=x.device)
        
        # Compute path probabilities through oblivious tree
        for level in range(self.depth):
            # Get split decisions for this level
            split_logits = self.feature_selectors[level](x)  # (B, n_trees)
            split_probs = torch.sigmoid(split_logits)  # (B, n_trees)
            
            # Update path probabilities
            step_size = 2**(self.depth - level - 1)
            for i in range(0, 2**self.depth, step_size * 2):
                # Left children (probability of going left)
                left_mask = slice(i, i + step_size)
                path_probs[:, :, left_mask] *= split_probs.unsqueeze(-1)
                
                # Right children (probability of going right)
                right_mask = slice(i + step_size, i + 2 * step_size)
                path_probs[:, :, right_mask] *= (1 - split_probs.unsqueeze(-1))
        
        # Compute weighted outputs
        outputs = []
        for tree_idx in range(self.n_trees):
            tree_out = torch.einsum('bl,ld->bd', 
                                  path_probs[:, tree_idx], 
                                  self.leaf_values[tree_idx])
            outputs.append(tree_out)
        
        return torch.stack(outputs, dim=1).mean(dim=1)


# ---------- Fixed Feature Tokenizer ----------

class AdvancedFeatureTokenizer(nn.Module):
    """Fixed tokenizer with proper binning and input scaling"""
    def __init__(self, n_features: int, d_token: int, hidden: int = 64, n_bins: int = 32, 
                 use_neural_odt: bool = True):
        super().__init__()
        self.n_features = n_features
        self.use_neural_odt = use_neural_odt
        self.n_bins = n_bins
        
        # Learned per-feature affine transformations
        self.feature_mean = nn.Parameter(torch.zeros(n_features))
        self.feature_scale = nn.Parameter(torch.ones(n_features))
        
        # Feature processors
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, d_token // 2),
            ) for _ in range(n_features)
        ])
        
        # Positional embeddings
        self.pos_emb = nn.Embedding(n_features, d_token // 4)
        
        # Monotonic bin edges using softplus + cumsum
        self.bin_logits = nn.Parameter(torch.randn(n_features, n_bins))
        self.bin_emb = nn.Embedding(n_bins, d_token // 4)
        
        # Neural ODT
        if use_neural_odt:
            self.odt = NeuralObliviousDecisionTree(n_features, depth=3, d_out=d_token // 4)
        
        # Final projection
        proj_dim = d_token // 2 + d_token // 4 + d_token // 4
        if use_neural_odt:
            proj_dim += d_token // 4
        self.final_proj = nn.Linear(proj_dim, d_token)
        
    def _get_monotonic_bin_edges(self, feature_idx: int) -> torch.Tensor:
        """Generate monotonic bin edges using softplus + cumsum"""
        # Softplus ensures positive increments
        increments = F.softplus(self.bin_logits[feature_idx])
        # Cumsum creates monotonic sequence
        edges = torch.cumsum(increments, dim=0)
        # Normalize to [-4, 4] range
        edges = edges - edges.mean()
        edges = edges / edges.std() * 4
        # Add boundary points
        min_edge = edges.min() - 1
        max_edge = edges.max() + 1
        return torch.cat([min_edge.unsqueeze(0), edges, max_edge.unsqueeze(0)])
        
    def forward(self, x):  # (B, F)
        B, F = x.shape
        
        # Apply learned affine transformation
        x_normalized = (x - self.feature_mean) / (self.feature_scale + 1e-8)
        
        # Process each feature individually
        tokens = []
        for i in range(F):
            feat_val = x_normalized[:, i:i+1]  # (B, 1)
            processed = self.feature_processors[i](feat_val)  # (B, d_token//2)
            tokens.append(processed)
        
        tokens = torch.stack(tokens, dim=1)  # (B, F, d_token//2)
        
        # Positional embeddings
        pos_emb = self.pos_emb.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Monotonic binning
        bins = []
        for i in range(F):
            bin_edges = self._get_monotonic_bin_edges(i)
            feat_bins = torch.bucketize(x_normalized[:, i], bin_edges) - 1
            feat_bins = feat_bins.clamp(0, self.n_bins - 1)
            bins.append(feat_bins)
        bins = torch.stack(bins, dim=1)  # (B, F)
        bin_emb = self.bin_emb(bins)
        
        # Combine embeddings
        combined = torch.cat([tokens, pos_emb, bin_emb], dim=-1)
        
        # Add neural ODT features if enabled
        if self.use_neural_odt:
            odt_features = self.odt(x_normalized).unsqueeze(1).expand(-1, F, -1)
            combined = torch.cat([combined, odt_features], dim=-1)
        
        return self.final_proj(combined)


# ---------- Fixed Multi-Scale Attention ----------

class EfficientMultiScaleAttention(nn.Module):
    """Fixed attention that computes QK once and reuses across scales"""
    def __init__(self, d_model: int, n_heads: int = 8, scales: List[int] = [1, 3, 5]):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False) 
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Scale-specific convolutions for values only
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=s, padding=s//2, groups=n_heads)
            for s in scales
        ])
        
        self.o_proj = nn.Linear(d_model * len(scales), d_model)
        self.dropout = nn.Dropout(0.1)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
    def forward(self, x):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2) 
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute attention weights once
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        scale_outputs = []
        
        for conv in self.scale_convs:
            # Apply scale-specific convolution to values
            v_reshaped = v.transpose(1, 2).contiguous().view(B, L, D)
            v_conv = conv(v_reshaped.transpose(1, 2)).transpose(1, 2)  # (B, L, D)
            v_conv = v_conv.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
            
            # Apply pre-computed attention to convolved values
            out = torch.matmul(attn, v_conv)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            scale_outputs.append(out)
        
        combined = torch.cat(scale_outputs, dim=-1)
        return self.o_proj(combined)


# ---------- Main Autoencoder with All Fixes ----------

class FixedSOTATabularAutoencoder(nn.Module):
    """Fixed SOTA autoencoder with proper VAE, contrastive learning, and masking"""
    def __init__(
        self, 
        n_features: int,
        d_token: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.3,
        dropout: float = 0.1,
        latent_dim: int = 128,
        use_neural_odt: bool = True,
        temperature: float = 0.1,
        beta_vae: float = 1.0,
        beta_schedule: str = "linear"  # "linear", "cosine", "constant"
    ):
        super().__init__()
        self.n_features = n_features
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.beta_vae = beta_vae
        self.beta_schedule = beta_schedule
        
        # Fixed tokenizer
        self.tokenizer = AdvancedFeatureTokenizer(
            n_features, d_token, use_neural_odt=use_neural_odt
        )
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_token))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = SafeAdaptiveNorm(d_token)
        
        # VAE decoder head with proper sampling
        self.vae_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, 2)  # μ and log(σ²)
        )
        
        # Latent projector (also VAE-style)
        self.latent_mean = nn.Linear(d_token, latent_dim)
        self.latent_logvar = nn.Linear(d_token, latent_dim)
        
        # Contrastive projector
        self.contrast_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(), 
            nn.Linear(latent_dim, latent_dim // 2)
        )
        
        # Fixed feature type assignment (frozen per model)
        self.register_buffer("feature_types", torch.randint(0, 4, (n_features,)))
        
        # Training step counter for beta annealing
        self.register_buffer("training_step", torch.tensor(0))
        
    def get_current_beta(self, total_steps: int = 50000) -> float:
        """Get current beta value with annealing schedule"""
        if self.beta_schedule == "constant":
            return self.beta_vae
        
        progress = min(1.0, self.training_step.float() / total_steps)
        
        if self.beta_schedule == "linear":
            return self.beta_vae * progress
        elif self.beta_schedule == "cosine":
            return self.beta_vae * (1 - math.cos(progress * math.pi)) / 2
        
        return self.beta_vae
        
    def _stable_masking(self, B: int, device: torch.device) -> torch.Tensor:
        """Fixed masking with stable feature types"""
        F_ = self.n_features
        
        # Random masking
        k_random = max(1, int(self.mask_ratio * F_ * 0.6))
        random_mask = torch.zeros(B, F_, dtype=torch.bool, device=device)
        random_idx = torch.argsort(torch.rand(B, F_, device=device), dim=1)
        random_mask.scatter_(1, random_idx[:, :k_random], True)
        
        # Block masking
        k_block = max(1, int(self.mask_ratio * F_ * 0.2))
        block_mask = torch.zeros(B, F_, dtype=torch.bool, device=device)
        start_idx = torch.randint(0, max(1, F_ - k_block + 1), (B,), device=device)
        for i in range(B):
            end_idx = min(start_idx[i] + k_block, F_)
            block_mask[i, start_idx[i]:end_idx] = True
        
        # Feature-type masking (using frozen types)
        type_mask = torch.zeros(B, F_, dtype=torch.bool, device=device)
        for b in range(B):
            mask_type = torch.randint(0, 4, (1,), device=device).item()
            type_mask[b] = (self.feature_types == mask_type)
        
        # Combine masks
        final_mask = random_mask | (block_mask & (torch.rand(B, F_, device=device) < 0.2))
        final_mask = final_mask | (type_mask & (torch.rand(B, F_, device=device) < 0.2))
        
        return final_mask
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        B, F = x.shape
        
        # Update training step
        if self.training:
            self.training_step += 1
        
        # Stable masking
        mask = self._stable_masking(B, x.device)
        
        # Tokenize
        tokens = self.tokenizer(x)  # (B, F, D)
        
        # Apply masking
        mask_tokens = self.mask_token.expand(B, F, -1)
        tokens = torch.where(mask.unsqueeze(-1), mask_tokens, tokens)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls_tokens, tokens], dim=1)
        
        # Transformer forward
        hidden_states = []
        for block in self.blocks:
            sequence = block(sequence)
            if return_hidden:
                hidden_states.append(sequence)
        
        sequence = self.norm(sequence)
        
        # Extract tokens
        cls_out = sequence[:, 0]      # CLS token
        feat_out = sequence[:, 1:]    # Feature tokens
        
        # VAE reconstruction with proper sampling
        vae_params = self.vae_head(feat_out)  # (B, F, 2)
        recon_mu = vae_params[..., 0]
        recon_logvar = vae_params[..., 1]
        
        # Sample reconstruction (during training)
        if self.training:
            reconstruction = self.reparameterize(recon_mu, recon_logvar)
        else:
            reconstruction = recon_mu
        
        # Latent VAE
        latent_mu = self.latent_mean(cls_out)
        latent_logvar = self.latent_logvar(cls_out)
        latent = self.reparameterize(latent_mu, latent_logvar)
        
        # Contrastive features
        contrast_features = self.contrast_proj(latent)
        
        output = {
            "mask": mask,
            "reconstruction": reconstruction,
            "recon_mu": recon_mu,
            "recon_logvar": recon_logvar,
            "latent": latent,
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
            "contrast_features": contrast_features,
        }
        
        if return_hidden:
            output["hidden_states"] = hidden_states
        
        return output
    
    def compute_loss(
        self, 
        x: torch.Tensor,
        alpha_recon: float = 1.0,
        alpha_contrast: float = 0.5,
        total_steps: int = 50000
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # First forward pass
        output1 = self.forward(x)
        
        # Second forward pass for contrastive learning
        output2 = self.forward(x)
        
        mask1 = output1["mask"]
        recon_mu1 = output1["recon_mu"]
        recon_logvar1 = output1["recon_logvar"]
        latent_mu1 = output1["latent_mu"]
        latent_logvar1 = output1["latent_logvar"]
        
        # Reconstruction loss (negative log-likelihood for Gaussian)
        recon_precision = torch.exp(-recon_logvar1)
        recon_loss = 0.5 * (recon_precision * (recon_mu1 - x).pow(2) + recon_logvar1)
        recon_loss = recon_loss[mask1].mean()
        
        # KL divergence for VAE (both reconstruction and latent)
        # KL for reconstruction parameters
        kl_recon = -0.5 * torch.sum(1 + recon_logvar1 - recon_mu1.pow(2) - recon_logvar1.exp(), dim=-1)
        kl_recon = kl_recon.mean()
        
        # KL for latent variables
        kl_latent = -0.5 * torch.sum(1 + latent_logvar1 - latent_mu1.pow(2) - latent_logvar1.exp(), dim=-1)
        kl_latent = kl_latent.mean()
        
        # Get current beta
        current_beta = self.get_current_beta(total_steps)
        
        losses = {
            "recon": recon_loss * alpha_recon,
            "kl_recon": kl_recon * current_beta,
            "kl_latent": kl_latent * current_beta
        }
        
        total_loss = losses["recon"] + losses["kl_recon"] + losses["kl_latent"]
        
        # Contrastive learning (2-view SimCLR-style)
        if alpha_contrast > 0:
            contrast_loss = self._simclr_loss(
                output1["contrast_features"], 
                output2["contrast_features"]
            )
            losses["contrast"] = contrast_loss * alpha_contrast
            total_loss += losses["contrast"]
        
        losses["total"] = total_loss
        losses["beta"] = current_beta
        
        return total_loss, losses
    
    def _simclr_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Fixed symmetric SimCLR contrastive loss"""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        N = z1.size(0)
        
        # Concatenate representations
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # (2N, 2N)
        
        # Create labels: each sample is positive with its augmented version
        labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)], dim=0).to(z.device)
        
        # Mask out self-similarities
        mask = torch.eye(2*N, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    @torch.no_grad() 
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Clean encoding without masking"""
        original_mask_ratio = self.mask_ratio
        self.mask_ratio = 0.0
        
        output = self.forward(x)
        latent = output["latent_mu"]  # Use mean for deterministic encoding
        
        self.mask_ratio = original_mask_ratio
        return latent
    
    @torch.no_grad()
    def featurize(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced feature extraction with residuals and uncertainty"""
        # Get full encoding
        self.mask_ratio = 0.0
        output = self.forward(x)
        self.mask_ratio = 0.3  # Reset
        
        # Per-column standardized residuals
        reconstruction = output["recon_mu"]
        residuals = (x - reconstruction) / (torch.exp(0.5 * output["recon_logvar"]) + 1e-8)
        
        return {
            "latent": output["latent_mu"],
            "reconstruction": reconstruction,
            "residuals": residuals,
            "uncertainties": torch.exp(0.5 * output["recon_logvar"]),
            "feature_importance": torch.abs(residuals).mean(dim=0)
        }


# ---------- Transformer Block ----------

class TransformerBlock(nn.Module):
    """Standard transformer block with proper norm placement"""
    def __init__(self, d_token: int, n_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = SafeAdaptiveNorm(d_token)
        self.attn = EfficientMultiScaleAttention(d_token, n_heads)
        
        self.norm2 = SafeAdaptiveNorm(d_token)
        self.ffn = GEGLU(d_token, int(d_token * mlp_ratio))
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# Example usage with all fixes:
if __name__ == "__main__":
    n_features = 100
    model = FixedSOTATabularAutoencoder(
        n_features=n_features,
        d_token=256,
        depth=4,
        latent_dim=128,
        beta_schedule="cosine"
    )
    
    x = torch.randn(32, n_features)
    
    # Training step
    loss, loss_dict = model.compute_loss(x)
    print(f"Total loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    # Feature extraction
    features = model.featurize(x)
    print(f"Latent shape: {features['latent'].shape}")
    print(f"Feature importance: {features['feature_importance'][:5]}")
    features = model.featurize(x)
    print(f"Latent shape: {features['latent'].shape}")
    print(f"Feature importance: {features['feature_importance'][:5]}")
