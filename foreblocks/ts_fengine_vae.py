from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset


# === RoPE Positional Encoding ===
class RoPEPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        t = torch.arange(x.size(1), device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


# === SwiGLU FeedForward Block ===


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, bias=False):
        super().__init__()
        # Use bias=False for better performance (common in modern architectures)
        self.proj_in = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # Fused gating operation
        gate, up = self.proj_in(x).chunk(2, dim=-1)
        return self.proj_out(self.dropout(F.silu(gate) * up))


# === Flash Attention Compatible Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1, use_flash_attn=True):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)  # RMSNorm is faster than LayerNorm
        self.norm2 = nn.RMSNorm(d_model)

        # Use SDPA for potential Flash Attention acceleration
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            bias=False,  # Disable bias for better performance
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Increased FFN ratio for better capacity
        self.ffn = SwiGLU(d_model, int(d_model * 8 / 3), dropout, bias=False)
        self.pos = RoPEPositionalEncoding(d_model)

    def forward(self, x):
        # Apply RoPE before attention
        self.pos(x)

        # Pre-norm architecture with residual connections
        normed = self.norm1(x)
        attn_out = self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.dropout1(attn_out)

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# === Encoder ===
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, d_model=128, seq_len=24, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, dropout=dropout) for _ in range(2)]
        )
        self.flatten = nn.Flatten()
        self.to_mu = nn.Linear(d_model * seq_len, latent_dim)
        self.to_logvar = nn.Linear(d_model * seq_len, latent_dim)

        self._init_weights()

    def _init_weights(self):
        # Better weight initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)  # B x T x d_model
        x = self.blocks(x)
        flat = self.flatten(x)
        return self.to_mu(flat), self.to_logvar(flat)


# === Decoder ===
class TimeSeriesDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, seq_len, d_model=128, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, seq_len * d_model), nn.SiLU()
        )
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, dropout=dropout) for _ in range(2)]
        )
        self.out_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z):
        x = self.latent_proj(z).view(z.size(0), self.seq_len, -1)
        x = self.blocks(x)
        return self.out_proj(x)


# === VAE Wrapper ===
class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, seq_len, d_model=128):
        super().__init__()
        self.encoder = TimeSeriesEncoder(
            input_dim, latent_dim, d_model=d_model, seq_len=seq_len
        )
        self.decoder = TimeSeriesDecoder(
            latent_dim, input_dim, seq_len, d_model=d_model
        )
        self.seq_len = seq_len

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss_fn(self, x, recon, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss, kl


# === Trainer ===
class TimeSeriesVAETrainer:
    def __init__(self, model, device="cuda", use_amp=True, ema_decay=0.999):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device != "cpu"
        self.ema_decay = ema_decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.scaler = GradScaler() if self.use_amp else None

        self.model_ema = deepcopy(self.model.decoder).to(device)
        for p in self.model_ema.parameters():
            p.requires_grad = False

    def update_ema(self):
        for p_ema, p in zip(
            self.model_ema.parameters(), self.model.decoder.parameters()
        ):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def train(self, series, epochs=100, batch_size=128, kl_warmup_epochs=20):
        dataset = TensorDataset(torch.tensor(series, dtype=torch.float32))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=2,
            persistent_workers=True,
        )

        for epoch in range(epochs):
            self.model.train()
            total_loss, total_rec, total_kl = 0, 0, 0
            beta = min(1.0, epoch / kl_warmup_epochs)

            for (x,) in loader:
                x = x.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                with autocast(device_type="cuda") if self.use_amp else torch.no_grad():
                    recon, mu, logvar = self.model(x)
                    loss, rec_loss, kl_loss = self.model.loss_fn(
                        x, recon, mu, logvar, beta
                    )

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                self.update_ema()

                total_loss += loss.item()
                total_rec += rec_loss.item()
                total_kl += kl_loss.item()

            print(
                f"[{epoch:03d}] β={beta:.3f} | Loss: {total_loss:.4f} | Recon: {total_rec:.4f} | KL: {total_kl:.4f}"
            )

    @torch.no_grad()
    def generate(self, n_samples, use_ema=True):
        z = torch.randn(
            n_samples, self.model.encoder.to_mu.out_features, device=self.device
        )
        decoder = self.model_ema if use_ema else self.model.decoder
        decoder.eval()
        with autocast(device_type="cuda") if self.use_amp else torch.no_grad():
            return decoder(z).cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
