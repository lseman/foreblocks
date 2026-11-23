"""Rolling-origin Corona waveform forecast benchmark. Run with PYTHONPATH=src."""

import csv
import sys
import time
from contextlib import redirect_stdout

import numpy as np
import torch
from torch import nn

from foreblocks.nn.transformer import TransformerEncoder


class Forecast(nn.Module):
    def __init__(self, patching, gate, router):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_size=1,
            d_model=32,
            n_heads=4,
            num_layers=2,
            ff_dim=64,
            patching=patching,
            dropout=0.0,
            attention_options={
                "gated_attention": gate is not None,
                "gated_attention_mode": gate or "per_head",
            },
            moe_experts=4 if router else 0,
            moe_top_k=2,
            moe_options={
                "router_type": router,
                "router_aux_num_aux": 4,
                "use_bias": True,
                "moe_router_bias_balance": False,
            }
            if router
            else {},
        )
        self.head = nn.Linear(32, 8)

    def forward(self, x):
        result = self.encoder(x.unsqueeze(-1))
        return self.head(result.last_hidden_state[:, -1]), result.aux_loss


def measure(series, cutoff, lookback, name, gate=None, router=None):
    samples = (series - series[:cutoff].mean()) / series[:cutoff].std()
    windows = samples.unfold(0, lookback + 8, 1)
    train = windows[: cutoff - lookback - 7]
    test = windows[cutoff : cutoff + 16000 - lookback - 7 : 32]
    torch.manual_seed(42)
    if name == "persistence":

        def predict(x):
            return x[:, -1:].expand(-1, 8)
    elif name == "linear":
        # OLS with intercept, fitted to every available training window.
        train64 = train.double()
        design = nn.functional.pad(train64[:, :lookback], (0, 1), value=1)
        weights = torch.linalg.lstsq(design, train64[:, lookback:]).solution.float()

        def predict(x):
            return x @ weights[:-1] + weights[-1]
    else:
        with redirect_stdout(sys.stderr):
            model = Forecast(name, gate, router).to(series.device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        rng = torch.Generator().manual_seed(17)
        for _ in range(100):
            batch = train[torch.randint(len(train), (32,), generator=rng)]
            prediction, aux = model(batch[:, :lookback])
            loss = (prediction - batch[:, lookback:]).square().mean() + aux
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()

        def predict(x):
            return model(x)[0]

    with torch.inference_mode():
        outputs = torch.cat([predict(b[:, :lookback]) for b in test.split(32)])
        mae = (outputs - test[:, lookback:]).abs().mean(0)
        x = test[:32, :lookback].contiguous()
        for _ in range(5):
            predict(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            predict(x)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 20
    return [cutoff, lookback, router or gate or name, *mae.tolist(), latency]


def main():
    raw = np.loadtxt(
        "datasets/Corona 50KV.csv", delimiter=",", usecols=1, max_rows=100000
    )
    series = torch.tensor(raw, device="cuda", dtype=torch.float32)
    writer = csv.writer(sys.stdout)
    writer.writerow(["cutoff", "lookback", "model", *range(1, 9), "batch_ms"])
    for cutoff in (48000, 64000, 80000):
        for lookback in (64, 256):
            for name in ("persistence", "linear", "none", "shared"):
                writer.writerow(measure(series, cutoff, lookback, name))
            for gate in ("per_head", "query_head"):
                writer.writerow(measure(series, cutoff, lookback, "shared", gate=gate))
            for router in ("linear", "token_conditioned_auxiliary"):
                row = measure(series, cutoff, lookback, "shared", router=router)
                row[2] = "moe_" + router
                writer.writerow(row)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
