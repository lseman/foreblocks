import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ---------- Advanced Norms & Activations ----------
class LayerScale(nn.Module):
    """LayerScale from CaiT - helps with training stability"""
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        return x * self.gamma


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    
    def forward(self, x):
        norm = x.norm(keepdim=True, dim=-1) / (x.shape[-1] ** 0.5)
        return self.weight * x / (norm + self.eps)


class AdaptiveNorm(nn.Module):
    """Adaptive normalization that learns to choose between different norms"""
    def __init__(self, d):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d)
        self.rms_norm = RMSNorm(d)
        self.batch_norm = nn.BatchNorm1d(d)
        self.weight = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
    
    def forward(self, x):
        w = F.softmax(self.weight, dim=0)
        if x.dim() == 3:  # (B, L, D)
            ln_out = self.layer_norm(x)
            rms_out = self.rms_norm(x)
            bn_out = self.batch_norm(x.transpose(-1, -2)).transpose(-1, -2)
        else:  # (B, D)
            ln_out = self.layer_norm(x)
            rms_out = self.rms_norm(x)
            bn_out = self.batch_norm(x)
        return w[0] * ln_out + w[1] * rms_out + w[2] * bn_out


class GEGLU(nn.Module):
    """GEGLU activation - often performs better than SwiGLU"""
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden * 2)
        self.w2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        u, v = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.dropout(F.gelu(u) * v))


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ---------- Advanced Feature Processing ----------
class NeuralObliviousDecisionTree(nn.Module):
    """Neural ODT for tabular data - captures feature interactions"""
    def __init__(self, n_features, depth=4, n_trees=8, d_out=64):
        super().__init__()
        self.depth = depth
        self.n_trees = n_trees
        
        # Feature selection and splitting
        self.feature_selector = nn.ModuleList([
            nn.Linear(n_features, 2**depth) for _ in range(n_trees)
        ])
        
        # Leaf values
        self.leaf_values = nn.Parameter(torch.randn(n_trees, 2**depth, d_out))
        
    def forward(self, x):  # (B, F)
        B = x.size(0)
        outputs = []
        
        for tree_idx in range(self.n_trees):
            # Get decision values
            decisions = torch.sigmoid(self.feature_selector[tree_idx](x))  # (B, 2^depth)
            
            # Soft routing through tree
            leaf_probs = decisions / decisions.sum(dim=1, keepdim=True)
            
            # Weighted sum of leaf values
            tree_out = torch.einsum('bl,ld->bd', leaf_probs, self.leaf_values[tree_idx])
            outputs.append(tree_out)
        
        return torch.stack(outputs, dim=1).mean(dim=1)  # Average over trees


class AdvancedFeatureTokenizer(nn.Module):
    """Enhanced tokenizer with multiple advanced techniques"""
    def __init__(self, n_features: int, d_token: int, hidden: int = 64, n_bins: int = 32, 
                 use_neural_odt: bool = True):
        super().__init__()
        self.n_features = n_features
        self.use_neural_odt = use_neural_odt
        
        # Learnable per-feature transformations
        self.feature_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                Mish(),
                nn.Dropout(0.1),
                nn.Linear(hidden, d_token // 2),
            ) for _ in range(n_features)
        ])
        
        # Learned positional embeddings for features
        self.pos_emb = nn.Embedding(n_features, d_token // 4)
        
        # Adaptive binning with learned boundaries
        self.register_buffer("bin_edges", torch.linspace(-4, 4, n_bins+1))
        self.bin_emb = nn.Embedding(n_bins, d_token // 4)
        self.bin_adjustment = nn.Parameter(torch.zeros(n_features, n_bins+1))
        
        # Neural ODT for capturing interactions
        if use_neural_odt:
            self.odt = NeuralObliviousDecisionTree(n_features, depth=3, d_out=d_token // 4)
        
        # Final projection
        proj_dim = d_token // 2 + d_token // 4 + d_token // 4
        if use_neural_odt:
            proj_dim += d_token // 4
        self.final_proj = nn.Linear(proj_dim, d_token)
        
    def forward(self, x):  # (B, F)
        B, F = x.shape
        tokens = []
        
        # Process each feature individually
        for i in range(F):
            feat_val = x[:, i:i+1]  # (B, 1)
            processed = self.feature_processors[i](feat_val)  # (B, d_token//2)
            tokens.append(processed)
        
        tokens = torch.stack(tokens, dim=1)  # (B, F, d_token//2)
        
        # Positional embeddings
        pos_emb = self.pos_emb.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Adaptive binning
        adjusted_edges = self.bin_edges.unsqueeze(0) + self.bin_adjustment
        bins = []
        for i in range(F):
            feat_bins = torch.bucketize(x[:, i], adjusted_edges[i]) - 1
            feat_bins = feat_bins.clamp(0, self.bin_emb.num_embeddings-1)
            bins.append(feat_bins)
        bins = torch.stack(bins, dim=1)  # (B, F)
        bin_emb = self.bin_emb(bins)
        
        # Combine embeddings
        combined = torch.cat([tokens, pos_emb, bin_emb], dim=-1)
        
        # Add neural ODT features if enabled
        if self.use_neural_odt:
            odt_features = self.odt(x).unsqueeze(1).expand(-1, F, -1)
            combined = torch.cat([combined, odt_features], dim=-1)
        
        return self.final_proj(combined)

class MultiScaleAttention(nn.Module):
    """Multi-scale attention with grouped convolutions - ALTERNATIVE VERSION"""
    def __init__(self, d_model: int, n_heads: int = 8, scales: List[int] = [1, 3, 5]):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False) 
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Scale-specific projections with proper grouping
        self.scale_projs = nn.ModuleList([
            nn.Conv1d(self.d_head, self.d_head, kernel_size=s, padding=s//2, groups=1)
            for s in scales
        ])
        
        self.o_proj = nn.Linear(d_model * len(scales), d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2) 
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        
        scale_outputs = []
        
        for scale_proj in self.scale_projs:
            # Apply convolution to each head separately
            v_scaled_heads = []
            for head in range(self.n_heads):
                v_head = v[:, head, :, :].transpose(1, 2)  # (B, d_head, L)
                v_head_scaled = scale_proj(v_head)  # (B, d_head, L)
                v_head_scaled = v_head_scaled.transpose(1, 2)  # (B, L, d_head)
                v_scaled_heads.append(v_head_scaled)
            
            v_scaled = torch.stack(v_scaled_heads, dim=1)  # (B, n_heads, L, d_head)
            
            # Standard attention
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, v_scaled)
            out = out.transpose(1, 2).contiguous().view(B, L, D)
            scale_outputs.append(out)
        
        combined = torch.cat(scale_outputs, dim=-1)
        return self.o_proj(combined)
class TabularTransformerBlock(nn.Module):
    """Enhanced transformer block with tabular-specific optimizations"""
    def __init__(self, d_token: int, n_heads: int = 8, mlp_ratio: float = 4.0, 
                 drop: float = 0.1, droppath: float = 0.1):
        super().__init__()
        
        self.norm1 = AdaptiveNorm(d_token)
        self.attn = MultiScaleAttention(d_token, n_heads)
        self.ls1 = LayerScale(d_token)
        self.drop_path1 = nn.Dropout(droppath) if droppath > 0 else nn.Identity()
        
        self.norm2 = AdaptiveNorm(d_token)
        self.ffn = GEGLU(d_token, int(d_token * mlp_ratio))
        self.ls2 = LayerScale(d_token)
        self.drop_path2 = nn.Dropout(droppath) if droppath > 0 else nn.Identity()
        
        # Feature interaction module
        self.feat_interact = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GLU(dim=-1),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # Attention with residual
        attn_out = self.attn(self.norm1(x))
        x = x + self.drop_path1(self.ls1(attn_out))
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop_path2(self.ls2(ffn_out))
        
        # Feature interaction
        interact_out = self.feat_interact(x)
        x = x + 0.1 * interact_out  # Smaller residual weight
        
        return x


# ---------- Enhanced Autoencoder with SOTA techniques ----------
class SOTATabularAutoencoder(nn.Module):
    """State-of-the-art tabular autoencoder with multiple advanced techniques"""
    def __init__(
        self, 
        n_features: int,
        d_token: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.3,
        dropout: float = 0.1,
        droppath: float = 0.2,
        latent_dim: int = 128,
        use_neural_odt: bool = True,
        use_spectral_norm: bool = True,
        temperature: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        
        # Advanced tokenizer
        self.tokenizer = AdvancedFeatureTokenizer(
            n_features, d_token, use_neural_odt=use_neural_odt
        )
        
        # Multiple special tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_token))
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02) 
        nn.init.trunc_normal_(self.sep_token, std=0.02)
        
        # Encoder blocks with different depths
        self.early_blocks = nn.ModuleList([
            TabularTransformerBlock(d_token, n_heads, mlp_ratio, dropout, droppath * (i / depth))
            for i in range(depth // 2)
        ])
        
        self.late_blocks = nn.ModuleList([
            TabularTransformerBlock(d_token, n_heads, mlp_ratio, dropout, droppath * ((i + depth//2) / depth))
            for i in range(depth - depth // 2)
        ])
        
        self.norm = AdaptiveNorm(d_token)
        
        # Multi-task decoder heads
        decoder_dim = d_token // 2
        
        # Reconstruction head with uncertainty
        self.recon_head = nn.Sequential(
            nn.Linear(d_token, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, 2)  # mean and log_var
        )
        
        # Latent projector with hierarchical structure
        self.latent_projector = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, latent_dim * 2),  # mean and log_var for VAE
            nn.GLU(dim=-1)
        )
        
        # Contrastive projector
        self.contrast_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(), 
            nn.Linear(latent_dim, latent_dim // 2)
        )
        
        # Optional spectral normalization
        if use_spectral_norm:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    spectral_norm(module)
        
        # Learnable mask strategy
        self.mask_strategy = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  # random, block, feature-type
        
    def _advanced_masking(self, B: int, device: torch.device) -> torch.Tensor:
        """Advanced masking with multiple strategies"""
        F_ = self.n_features
        strategy_weights = F.softmax(self.mask_strategy, dim=0)
        
        masks = []
        
        # Random masking
        k_random = max(1, int(self.mask_ratio * F_ * strategy_weights[0]))
        random_mask = torch.zeros(B, F_, dtype=torch.bool, device=device)
        random_idx = torch.argsort(torch.rand(B, F_, device=device), dim=1)
        random_mask.scatter_(1, random_idx[:, :k_random], True)
        masks.append(random_mask)
        
        # Block masking (consecutive features)
        k_block = max(1, int(self.mask_ratio * F_ * strategy_weights[1]))
        block_mask = torch.zeros(B, F_, dtype=torch.bool, device=device)
        start_idx = torch.randint(0, F_ - k_block + 1, (B,), device=device)
        for i in range(B):
            block_mask[i, start_idx[i]:start_idx[i] + k_block] = True
        masks.append(block_mask)
        
        # Feature-type masking (simulate different data types)
        type_mask = torch.zeros(B, F_, dtype=torch.bool, device=device)
        num_types = 4
        feature_types = torch.randint(0, num_types, (F_,), device=device)
        for b in range(B):
            mask_type = torch.randint(0, num_types, (1,), device=device).item()
            type_mask[b] = (feature_types == mask_type)
        masks.append(type_mask)
        
        # Combine masks
        final_mask = torch.zeros_like(masks[0])
        for i, mask in enumerate(masks):
            weight = strategy_weights[i]
            final_mask = final_mask | (torch.rand(B, F_, device=device) < weight) & mask
        
        return final_mask
    
    def forward(self, x: torch.Tensor, return_hidden: bool = False) -> Dict[str, torch.Tensor]:
        B, F = x.shape
        
        # Advanced masking
        mask = self._advanced_masking(B, x.device)
        
        # Tokenize
        tokens = self.tokenizer(x)  # (B, F, D)
        
        # Apply masking with learned mask token
        mask_tokens = self.mask_token.expand(B, F, -1)
        tokens = torch.where(mask.unsqueeze(-1), mask_tokens, tokens)
        
        # Add special tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sep_tokens = self.sep_token.expand(B, -1, -1)
        
        # Sequence: [CLS] features [SEP]
        sequence = torch.cat([cls_tokens, tokens, sep_tokens], dim=1)
        
        # Early blocks
        hidden_states = []
        for block in self.early_blocks:
            sequence = block(sequence)
            hidden_states.append(sequence)
        
        # Late blocks  
        for block in self.late_blocks:
            sequence = block(sequence)
            hidden_states.append(sequence)
        
        sequence = self.norm(sequence)
        
        # Extract tokens
        cls_out = sequence[:, 0]      # CLS token
        feat_out = sequence[:, 1:-1]  # Feature tokens
        
        # Reconstruction with uncertainty
        recon_params = self.recon_head(feat_out)  # (B, F, 2)
        recon_mean = recon_params[..., 0]
        recon_log_var = recon_params[..., 1]
        
        # Latent representation
        latent = self.latent_projector(cls_out)
        
        # Contrastive features
        contrast_features = self.contrast_proj(latent)
        
        output = {
            "mask": mask,
            "recon_mean": recon_mean,
            "recon_log_var": recon_log_var,
            "latent": latent,
            "contrast_features": contrast_features,
        }
        
        if return_hidden:
            output["hidden_states"] = hidden_states
        
        return output
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        y: Optional[torch.Tensor] = None,
        alpha_recon: float = 1.0,
        alpha_contrast: float = 0.5,
        alpha_reg: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        output = self.forward(x)
        mask = output["mask"]
        recon_mean = output["recon_mean"]
        recon_log_var = output["recon_log_var"]
        
        # Reconstruction loss with uncertainty
        recon_precision = torch.exp(-recon_log_var)
        recon_loss = 0.5 * (recon_precision * (recon_mean - x).pow(2) + recon_log_var)
        recon_loss = recon_loss[mask].mean()
        
        losses = {"recon": recon_loss * alpha_recon}
        total_loss = losses["recon"]
        
        # Contrastive learning
        if alpha_contrast > 0:
            # Create augmented view with different masking
            with torch.no_grad():
                output_aug = self.forward(x)
            
            contrast_loss = self._contrastive_loss(
                output["contrast_features"], 
                output_aug["contrast_features"]
            )
            losses["contrast"] = contrast_loss * alpha_contrast
            total_loss += losses["contrast"]
        
        # Regularization
        if alpha_reg > 0:
            # Feature diversity loss
            latent_std = torch.std(output["latent"], dim=0).mean()
            reg_loss = -torch.log(latent_std + 1e-8)  # Encourage diversity
            
            # Sparsity loss on attention weights (if available)
            losses["reg"] = reg_loss * alpha_reg
            total_loss += losses["reg"]
        
        losses["total"] = total_loss
        return total_loss, losses
    
    def _contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss"""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Positive pairs
        pos_sim = torch.sum(z1 * z2, dim=-1) / self.temperature
        
        # Negative pairs (within batch)
        neg_sim = torch.mm(z1, z2.T) / self.temperature
        
        # Remove diagonal (positive pairs)
        mask = torch.eye(z1.size(0), device=z1.device).bool()
        neg_sim = neg_sim.masked_fill(mask, float('-inf'))
        
        # Compute InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(z1.size(0), dtype=torch.long, device=z1.device)
        
        return F.cross_entropy(logits, labels)
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representations without masking"""
        original_mask_ratio = self.mask_ratio
        self.mask_ratio = 0.0  # No masking for encoding
        
        output = self.forward(x)
        latent = output["latent"]
        
        self.mask_ratio = original_mask_ratio
        return latent
        latent = output["latent"]
        
        self.mask_ratio = original_mask_ratio
        return latent
        
        self.mask_ratio = original_mask_ratio
        return latent
        
        self.mask_ratio = original_mask_ratio
        return latent
        
        self.mask_ratio = original_mask_ratio
        return latent
