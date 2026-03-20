import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


@dataclass
class HParams:
    d_model: int
    nhead: int
    nlayer: int
    ff_mult: int
    seq_len: int
    batch_size: int
    lr: float
    max_steps: int


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, hp: HParams):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hp.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, hp.seq_len, hp.d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hp.d_model,
            nhead=hp.nhead,
            dim_feedforward=hp.d_model * hp.ff_mult,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=hp.nlayer)
        self.ln = nn.LayerNorm(hp.d_model)
        self.head = nn.Linear(hp.d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(idx) + self.pos_emb[:, : idx.size(1), :]
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x)


class PrivacyController:
    def __init__(self, max_grad_norm: float = 0.9, delta: float = 1e-5):
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.activations = 0
        self.records = []

    def enforce(self, model: nn.Module, step: int, loss_value: float) -> dict:
        sq_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sq_sum += float(torch.sum(p.grad.detach() ** 2).cpu())
        grad_norm = math.sqrt(max(sq_sum, 1e-12))
        scale = 1.0
        activated = grad_norm > self.max_grad_norm
        if activated:
            scale = self.max_grad_norm / (grad_norm + 1e-12)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            self.activations += 1

        record = {
            "step": step,
            "loss": float(loss_value),
            "grad_norm": grad_norm,
            "clip_scale": float(scale),
            "activated": bool(activated),
        }
        self.records.append(record)
        return record

    def pac_style_bound(self, empirical_risk: float, n_samples: int, complexity: float) -> float:
        if n_samples <= 1:
            raise ValueError("Need at least 2 samples for bound computation")
        rad = math.sqrt((complexity + math.log(2.0 / self.delta)) / (2.0 * (n_samples - 1)))
        return float(empirical_risk + rad)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sample_batch(vocab_size: int, batch_size: int, seq_len: int):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return x, y


def run_once(hp: HParams, out_dir: Path, time_budget_s: float = 110.0):
    set_seed(1234)
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 2)))
    device = torch.device("cpu")

    vocab_size = 256
    model = TinyTransformer(vocab_size, hp).to(device)
    n_params = count_params(model)
    if n_params > 10_000_000:
        raise RuntimeError(f"Model too large ({n_params} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr)
    controller = PrivacyController(max_grad_norm=0.9)
    process = psutil.Process(os.getpid())

    losses = []
    mem_metrics = []
    instability = []

    start = time.time()
    n_samples = 0

    for step in range(hp.max_steps):
        if time.time() - start > time_budget_s:
            break

        x, y = sample_batch(vocab_size, hp.batch_size, hp.seq_len)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if not finite_tensor(logits):
            instability.append(f"step={step}: non-finite logits")
            raise FloatingPointError(instability[-1])

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        if not torch.isfinite(loss):
            instability.append(f"step={step}: non-finite loss")
            raise FloatingPointError(instability[-1])

        loss.backward()

        has_bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all().item():
                has_bad_grad = True
                break
        if has_bad_grad:
            instability.append(f"step={step}: non-finite gradients")
            raise FloatingPointError(instability[-1])

        controller.enforce(model, step=step, loss_value=float(loss.detach().cpu()))

        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        n_samples += hp.batch_size

        rss_mb = process.memory_info().rss / (1024 * 1024)
        mem_metrics.append({"step": step, "rss_mb": rss_mb, "elapsed_s": time.time() - start})

        if rss_mb > 1500:
            raise MemoryError(f"memory pressure detected at {rss_mb:.1f} MB")

        if step >= 4 and losses[-1] > losses[0] * 4:
            instability.append(f"step={step}: divergence (loss exploded)")
            raise FloatingPointError(instability[-1])

    if not losses:
        raise RuntimeError("No training steps executed")

    complexity = float(sum((p.detach().float().pow(2).sum().item() for p in model.parameters())))
    empirical_risk = float(np.mean(losses))
    bound = controller.pac_style_bound(empirical_risk, n_samples=n_samples, complexity=math.log1p(complexity))

    if not math.isfinite(bound):
        raise FloatingPointError("PAC bound is non-finite")
    if controller.activations == 0:
        raise RuntimeError("Privacy controller did not activate")

    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "training_losses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss"])
        writer.writeheader()
        for i, l in enumerate(losses):
            writer.writerow({"step": i, "loss": l})

    with (out_dir / "enforcement_log.json").open("w") as f:
        json.dump(controller.records, f, indent=2)

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(
            {
                "runtime_s": time.time() - start,
                "max_rss_mb": max(m["rss_mb"] for m in mem_metrics),
                "param_count": n_params,
                "steps": len(losses),
                "instability_events": instability,
            },
            f,
            indent=2,
        )

    with (out_dir / "memory_runtime.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "rss_mb", "elapsed_s"])
        writer.writeheader()
        writer.writerows(mem_metrics)

    with (out_dir / "pac_bound.txt").open("w") as f:
        f.write(f"{bound:.8f}\n")

    print(f"SUCCESS: bound={bound:.6f}, activations={controller.activations}, params={n_params}")


def main():
    out = Path("artifacts")
    configs = [
        HParams(d_model=192, nhead=6, nlayer=4, ff_mult=4, seq_len=96, batch_size=16, lr=2e-3, max_steps=30),
        HParams(d_model=160, nhead=5, nlayer=3, ff_mult=3, seq_len=72, batch_size=12, lr=1e-3, max_steps=24),
        HParams(d_model=128, nhead=4, nlayer=3, ff_mult=3, seq_len=64, batch_size=8, lr=8e-4, max_steps=20),
        HParams(d_model=96, nhead=4, nlayer=2, ff_mult=2, seq_len=48, batch_size=6, lr=5e-4, max_steps=16),
    ]

    last_error = None
    for i, cfg in enumerate(configs, start=1):
        print(f"Attempt {i}/{len(configs)} with cfg={cfg}")
        try:
            run_once(cfg, out)
            print("Training run completed.")
            return
        except (MemoryError, FloatingPointError, RuntimeError, ValueError) as exc:
            last_error = exc
            print(f"Attempt {i} failed: {exc}")
            if i < len(configs):
                print("Retrying with safer hyperparameters...")

    raise SystemExit(f"Model diverged or bound failed after retries: {last_error}")


if __name__ == "__main__":
    main()
