
from copy import deepcopy

# =============== GAN FOR SIGNAL AUGMENTATION ===============
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset


# ==== Rotary Position Embedding ====
class RoPEPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        seq_len = seq_len or x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin


# ==== Linear Attention ====
class LinearAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1, eps=1e-6):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.scale = self.d_k**-0.5
        self.dropout = nn.Dropout(dropout)
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _phi(self, x):
        return F.elu(x) + 1

    def forward(self, x):
        B, T, _ = x.size()
        q = self._phi(self.q_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2))
        k = self._phi(self.k_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2))
        v = self.v_proj(x).view(B, T, self.nhead, self.d_k).transpose(1, 2)

        kv = torch.einsum("bhtd,bhtm->bhdm", k, v)
        z = 1 / (torch.einsum("bhtd,bhd->bht", q, k.sum(dim=2) + self.eps) + self.eps)
        out = torch.einsum("bhtd,bhdm,bht->bhtm", q, kv, z)
        return self.out_proj(self.dropout(out.transpose(1, 2).reshape(B, T, -1)))


# ==== Transformer Block ====
class OptimizedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = LinearAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# ==== Generator ====
class TimeSeriesGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, seq_len, d_model=32, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len, self.d_model = seq_len, d_model

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, seq_len * d_model)
        )

        self.pos_encoder = RoPEPositionalEncoding(d_model, seq_len)
        self.blocks = nn.ModuleList([OptimizedTransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, z):
        x = self.latent_proj(z).view(z.size(0), self.seq_len, self.d_model)
        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x)
        return self.output_proj(self.norm(x))


# ==== Discriminator ====
class MinibatchStdDev(nn.Module):
    def forward(self, x):
        std = x.std(dim=0, keepdim=True).mean(dim=1, keepdim=True)
        return torch.cat([x, std.expand(x.size(0), 1, x.size(2))], dim=1)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dilation=1):
        super().__init__()
        pad = (k - 1) * dilation
        self.depthwise = spectral_norm(nn.Conv1d(in_ch, in_ch, k, padding=pad, dilation=dilation, groups=in_ch))
        self.pointwise = spectral_norm(nn.Conv1d(in_ch, out_ch, 1))
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        if self.depthwise.padding[0] > 0:
            x = x[:, :, :-self.depthwise.padding[0]]
        return self.act(self.norm(self.pointwise(x)))


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        self.stem = spectral_norm(nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3))
        self.backbone = nn.Sequential(*[
            l for i in range(num_layers)
            for l in (
                [DepthwiseSeparableConv1d(hidden_dim, hidden_dim, k=3, dilation=2**min(i, 4))]
                + ([nn.AdaptiveAvgPool1d(64)] if i % 2 == 1 else [])
            )
        ])

        self.stddev = MinibatchStdDev()
        self.attn = nn.MultiheadAttention(hidden_dim + 1, 4, batch_first=True) if use_attention else None
        self.attn_norm = nn.LayerNorm(hidden_dim + 1) if use_attention else None
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim + 1, hidden_dim // 2)), nn.SiLU(),
            spectral_norm(nn.Linear(hidden_dim // 2, 1))
        )

    def forward(self, x):
        x = F.silu(self.stem(x.transpose(1, 2)))
        x = self.backbone(x)
        x = self.stddev(x).transpose(1, 2)

        if self.use_attention:
            x = self.attn_norm(x)
            x, _ = self.attn(x, x, x)

        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.classifier(x)


# ==== Trainer ====
class TimeSeriesWGAN_GP:
    def __init__(self, input_dim, latent_dim, seq_len, hidden_dim=64, lambda_gp=10.0, n_critic=1, device="cuda", use_amp=True):
        self.device = device
        self.seq_len, self.input_dim, self.latent_dim = seq_len, input_dim, latent_dim
        self.lambda_gp, self.n_critic, self.use_amp = lambda_gp, n_critic, use_amp and device != "cpu"

        self.G = TimeSeriesGenerator(latent_dim, input_dim, seq_len, d_model=128, nhead=4).to(device)
        self.D = TimeSeriesDiscriminator(input_dim, hidden_dim).to(device)

        self.opt_G = optim.AdamW(self.G.parameters(), lr=1e-4, betas=(0.0, 0.9), weight_decay=1e-4)
        self.opt_D = optim.AdamW(self.D.parameters(), lr=4e-4, betas=(0.0, 0.9), weight_decay=1e-4)
        self.sched_G = optim.lr_scheduler.StepLR(self.opt_G, step_size=200, gamma=0.8)
        self.sched_D = optim.lr_scheduler.StepLR(self.opt_D, step_size=200, gamma=0.8)

        self.scaler = GradScaler(device) if self.use_amp else None
        self.G_ema = deepcopy(self.G)
        self.ema_decay = 0.999

    def _sample_noise(self, N): return torch.randn(N, self.latent_dim, device=self.device)

    @torch.no_grad()
    def update_ema(self):
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def compute_r1_penalty(self, real, gamma=10.0):
        real.requires_grad_(True)
        with autocast(device_type="cuda" if self.device == "cuda" else "cpu") if self.use_amp else torch.no_grad():
            scores = self.D(real)
        grads = torch.autograd.grad(scores.sum(), real, create_graph=True)[0]
        return (gamma / 2) * grads.view(grads.size(0), -1).pow(2).sum(1).mean()

    def train(self, real_series, epochs=1000, batch_size=256, warmup_epochs=50):
        loader = DataLoader(
            TensorDataset(torch.tensor(real_series, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True, drop_last=True,
            pin_memory=True, num_workers=2, persistent_workers=True,
        )

        for epoch in range(epochs):
            d_loss_acc, g_loss_acc, n_batches = 0.0, 0.0, 0
            for real_batch, in loader:
                real_batch = real_batch.to(self.device, non_blocking=True)
                z = self._sample_noise(real_batch.size(0))

                # === Train Discriminator ===
                for _ in range(self.n_critic):
                    with autocast(device_type="cuda" if self.device == "cuda" else "cpu") if self.use_amp else torch.no_grad():
                        fake = self.G(z).detach()
                        d_real = self.D(real_batch).mean()
                        d_fake = self.D(fake).mean()
                        gamma = 5.0 if epoch < warmup_epochs else 10.0
                        r1 = self.compute_r1_penalty(real_batch, gamma)
                        d_loss = d_fake - d_real + r1

                    self.opt_D.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(d_loss).backward()
                        self.scaler.unscale_(self.opt_D)
                    else:
                        d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                    (self.scaler.step(self.opt_D) if self.use_amp else self.opt_D.step())
                    if self.use_amp: self.scaler.update()


                # === Train Generator ===
                with autocast(device_type="cuda" if self.device == "cuda" else "cpu") if self.use_amp else torch.no_grad():
                    fake = self.G(z)
                    g_loss = -self.D(fake).mean()

                self.opt_G.zero_grad()
                if self.use_amp:
                    self.scaler.scale(g_loss).backward()
                    self.scaler.unscale_(self.opt_G)
                else:
                    g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
                (self.scaler.step(self.opt_G) if self.use_amp else self.opt_G.step())
                if self.use_amp: self.scaler.update()
                self.update_ema()


                d_loss_acc += d_loss.item()
                g_loss_acc += g_loss.item()
                n_batches += 1

            if epoch % 100 == 0:
                print(f"Epoch {epoch:04d} | D_loss: {d_loss_acc/n_batches:.4f} | G_loss: {g_loss_acc/n_batches:.4f}")

            self.sched_D.step()
            self.sched_G.step()

    @torch.no_grad()
    def generate(self, n_samples, use_ema=True):
        model = self.G_ema if use_ema else self.G
        model.eval()
        z = self._sample_noise(n_samples)
        with autocast(device_type="cuda") if self.use_amp else torch.no_grad():
            out = model(z)
        model.train()
        return out.cpu().numpy()

    def save_checkpoint(self, path):
        torch.save({
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "opt_G_state_dict": self.opt_G.state_dict(),
            "opt_D_state_dict": self.opt_D.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["G_state_dict"])
        self.D.load_state_dict(ckpt["D_state_dict"])
        self.opt_G.load_state_dict(ckpt["opt_G_state_dict"])
        self.opt_D.load_state_dict(ckpt["opt_D_state_dict"])
