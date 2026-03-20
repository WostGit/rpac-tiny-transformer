import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=128, d_model=128, n_heads=4, n_layers=2, seq_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        x = self.token_emb(idx) + self.pos_emb[:, : idx.size(1), :]
        x = self.encoder(x)
        x = self.ln(x)
        return self.head(x)


def estimate_params(cfg):
    d_model = cfg["d_model"]
    vocab = cfg["vocab_size"]
    layers = cfg["n_layers"]
    ff = d_model * 4
    embedding = vocab * d_model + cfg["seq_len"] * d_model
    per_layer = 4 * d_model * d_model + 2 * d_model * ff
    head = d_model * vocab
    return embedding + layers * per_layer + head


def compute_pac_bound(clipped_norms, n, delta=0.05):
    if n <= 0:
        raise RuntimeError("Bound cannot be computed: n must be positive.")
    empirical = float(np.mean(clipped_norms))
    if not np.isfinite(empirical):
        raise RuntimeError("Bound cannot be computed: empirical value is non-finite.")
    bound = empirical + math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    if not math.isfinite(bound):
        raise RuntimeError("Bound cannot be computed: PAC bound is non-finite.")
    return bound


def run_attempt(cfg, attempt_id):
    device = torch.device("cpu")
    torch.manual_seed(7 + attempt_id)
    np.random.seed(7 + attempt_id)

    model = TinyTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        seq_len=cfg["seq_len"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    if not (1_000_000 <= params <= 10_000_000):
        raise RuntimeError(f"Model size outside 1M-10M range: {params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    process = psutil.Process(os.getpid())
    peak_rss_mb = 0.0
    loss_curve = []
    enforce_log = []
    clipped_norms = []
    privacy_activations = 0

    for step in range(cfg["steps"]):
        x = torch.randint(0, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]), device=device)
        y = torch.randint(0, cfg["vocab_size"], (cfg["batch_size"], cfg["seq_len"]), device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Numerical instability: logits contain NaN/Inf.")

        loss = criterion(logits.reshape(-1, cfg["vocab_size"]), y.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("Numerical instability: loss is NaN/Inf.")

        loss.backward()

        grad_norm_sq = 0.0
        has_non_finite_grad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                has_non_finite_grad = True
                break
            grad_norm_sq += float(torch.sum(p.grad * p.grad).item())
        if has_non_finite_grad:
            raise FloatingPointError("Numerical instability: gradients contain NaN/Inf.")

        grad_norm = math.sqrt(max(grad_norm_sq, 0.0))
        max_grad_norm = cfg["max_grad_norm"]
        clip_coef = max_grad_norm / (grad_norm + 1e-8)
        activated = clip_coef < 1.0
        if activated:
            privacy_activations += 1
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        clipped_norms.append(min(grad_norm, max_grad_norm))
        enforce_log.append(
            {
                "step": step,
                "raw_grad_norm": grad_norm,
                "clip_threshold": max_grad_norm,
                "activated": activated,
            }
        )

        optimizer.step()
        loss_curve.append(float(loss.item()))

        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024 ** 2))
        if peak_rss_mb > cfg["memory_soft_limit_mb"]:
            raise MemoryError(f"Memory pressure detected at {peak_rss_mb:.1f} MB")

    if privacy_activations == 0:
        raise RuntimeError("Privacy controller did not activate during training.")

    bound = compute_pac_bound(clipped_norms, n=len(clipped_norms), delta=cfg["delta"])
    if not math.isfinite(bound):
        raise RuntimeError("Bound cannot be computed: value is non-finite.")

    return loss_curve, enforce_log, {
        "attempt": attempt_id,
        "params": params,
        "peak_rss_mb": peak_rss_mb,
        "runtime_sec": None,
        "final_loss": loss_curve[-1],
        "privacy_activations": privacy_activations,
        "pac_bound": bound,
        "config": cfg,
    }


def safer(cfg):
    new = dict(cfg)
    new["d_model"] = max(64, int(new["d_model"] * 0.75))
    new["n_layers"] = max(1, new["n_layers"] - 1)
    new["batch_size"] = max(2, new["batch_size"] // 2)
    new["seq_len"] = max(16, new["seq_len"] // 2)
    new["lr"] = max(1e-4, new["lr"] * 0.5)
    new["max_grad_norm"] = max(0.1, new["max_grad_norm"] * 0.8)
    return new


def main():
    Path("artifacts").mkdir(exist_ok=True)
    start = time.time()

    cfg = {
        "vocab_size": 128,
        "d_model": 192,
        "n_heads": 6,
        "n_layers": 3,
        "seq_len": 96,
        "batch_size": 16,
        "steps": 12,
        "lr": 2e-3,
        "max_grad_norm": 0.45,
        "delta": 0.05,
        "memory_soft_limit_mb": 1800,
    }

    for try_id in range(1, 4):
        try:
            if estimate_params(cfg) < 1_000_000:
                cfg["d_model"] = max(cfg["d_model"], 192)
                cfg["n_layers"] = max(cfg["n_layers"], 3)
                cfg["seq_len"] = max(cfg["seq_len"], 96)

            attempt_start = time.time()
            losses, log, metrics = run_attempt(cfg, try_id)
            metrics["runtime_sec"] = time.time() - attempt_start

            with open("artifacts/enforcement_log.json", "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
            with open("artifacts/pac_bound.json", "w", encoding="utf-8") as f:
                json.dump({"pac_bound": metrics["pac_bound"], "delta": cfg["delta"]}, f, indent=2)
            with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            np.savetxt("artifacts/loss_curve.csv", np.array(losses), delimiter=",", header="loss", comments="")
            plt.figure(figsize=(6, 4))
            plt.plot(losses)
            plt.title("Tiny Transformer Training Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.tight_layout()
            plt.savefig("artifacts/loss_curve.png")

            total_runtime = time.time() - start
            if total_runtime > 120:
                raise RuntimeError(f"Training exceeded 2-minute budget ({total_runtime:.1f}s).")

            print("Training and PAC checks completed successfully.")
            print(json.dumps(metrics, indent=2))
            return

        except (FloatingPointError, MemoryError) as e:
            print(f"Attempt {try_id} failed due to instability/pressure: {e}")
            if try_id == 3:
                raise RuntimeError(
                    f"Model diverged after retries with safer hyperparameters. Last error: {e}"
                ) from e
            cfg = safer(cfg)
        except RuntimeError as e:
            msg = str(e)
            if any(k in msg.lower() for k in ["privacy controller", "bound cannot be computed", "diverged"]):
                raise
            if try_id == 3:
                raise
            print(f"Attempt {try_id} runtime issue: {e}; retrying with safer hyperparameters")
            cfg = safer(cfg)

    raise RuntimeError("Model diverged and retries exhausted.")


if __name__ == "__main__":
    main()
