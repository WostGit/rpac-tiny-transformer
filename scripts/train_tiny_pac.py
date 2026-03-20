#!/usr/bin/env python3
"""Short CPU-only tiny-transformer training with PAC-style privacy controller checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.set_num_threads(2)
torch.manual_seed(7)
np.random.seed(7)


@dataclass
class HParams:
    vocab_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    seq_len: int = 64
    batch_size: int = 32
    steps: int = 30
    lr: float = 3e-4
    clip_norm: float = 0.8
    epsilon: float = 2.5
    delta: float = 1e-5


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + y)
        z = self.ff(x)
        return self.ln2(x + z)


class TinyTransformer(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.tok = nn.Embedding(hp.vocab_size, hp.d_model)
        self.pos = nn.Parameter(torch.zeros(1, hp.seq_len, hp.d_model))
        self.blocks = nn.ModuleList([TinyBlock(hp.d_model, hp.n_heads) for _ in range(hp.n_layers)])
        self.ln = nn.LayerNorm(hp.d_model)
        self.head = nn.Linear(hp.d_model, hp.vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos[:, : idx.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x)


def estimate_memory_mb(hp: HParams) -> float:
    # rough CPU tensor memory estimate for params + activations, in MB
    param_scale = hp.n_layers * hp.d_model * hp.d_model * 12
    activation_scale = hp.batch_size * hp.seq_len * hp.d_model * hp.n_layers * 8
    total_floats = param_scale + activation_scale
    return total_floats * 4 / (1024 * 1024)


def safer_hparams(hp: HParams) -> HParams:
    hp.d_model = max(32, hp.d_model // 2)
    hp.seq_len = max(16, hp.seq_len // 2)
    hp.batch_size = max(4, hp.batch_size // 2)
    hp.n_heads = max(1, min(hp.n_heads, hp.d_model // 32))
    hp.lr = min(hp.lr, 2e-4)
    hp.clip_norm = min(hp.clip_norm, 0.5)
    hp.steps = max(12, hp.steps - 5)
    return hp


def apply_privacy_controller(model: nn.Module, clip_norm: float) -> bool:
    enforced = False
    for p in model.parameters():
        if p.grad is None:
            continue
        grad_norm = p.grad.data.norm(2)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            continue
        if grad_norm > clip_norm:
            p.grad.data.mul_(clip_norm / (grad_norm + 1e-9))
            enforced = True
    return enforced


def compute_pac_style_bound(losses: list[float], hp: HParams, n_params: int) -> float:
    if not losses:
        raise RuntimeError("bound cannot be computed: no losses recorded")
    empirical = float(np.mean(losses[-5:]))
    complexity = math.sqrt((math.log(max(n_params, 2)) + math.log(2 / hp.delta)) / (2 * max(len(losses), 1)))
    bound = empirical + complexity + 1.0 / max(hp.epsilon, 1e-6)
    if not np.isfinite(bound):
        raise RuntimeError("bound cannot be computed: non-finite value")
    return float(bound)


def has_numerical_instability(logits: torch.Tensor, loss: torch.Tensor, bound: float | None, model: nn.Module) -> bool:
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        return True
    if bound is not None and not np.isfinite(bound):
        return True
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return True
    return False


def run_training(hp: HParams, max_runtime_sec: int, memory_budget_mb: float, artifact_dir: Path) -> None:
    device = torch.device("cpu")
    start = time.perf_counter()
    process = psutil.Process()

    adjustments: list[str] = []
    retries = 0

    while retries < 4:
        predicted = estimate_memory_mb(hp)
        if predicted > memory_budget_mb:
            adjustments.append(
                f"memory pressure predicted ({predicted:.1f}MB > budget {memory_budget_mb:.1f}MB), reducing model"
            )
            hp = safer_hparams(hp)

        model = TinyTransformer(hp).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr)

        losses: list[float] = []
        enforce_events: list[dict[str, float | int | bool]] = []
        bounds: list[float] = []
        numerical_issue = False
        controller_activated = False

        try:
            for step in range(hp.steps):
                if (time.perf_counter() - start) > max_runtime_sec:
                    break

                x = torch.randint(0, hp.vocab_size, (hp.batch_size, hp.seq_len), device=device)
                y = torch.roll(x, shifts=-1, dims=1)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, hp.vocab_size), y.reshape(-1))
                loss.backward()

                activated = apply_privacy_controller(model, hp.clip_norm)
                controller_activated = controller_activated or activated
                bound_now = compute_pac_style_bound(losses + [float(loss.item())], hp, n_params)

                if has_numerical_instability(logits, loss, bound_now, model):
                    numerical_issue = True
                    adjustments.append(f"numerical instability at retry={retries}, step={step}; using safer hyperparameters")
                    break

                optimizer.step()

                losses.append(float(loss.item()))
                bounds.append(float(bound_now))
                enforce_events.append(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "bound": float(bound_now),
                        "privacy_enforced": bool(activated),
                    }
                )

            if not losses:
                raise RuntimeError("model diverged: no successful optimization steps")

            final_bound = compute_pac_style_bound(losses, hp, n_params)
            if not controller_activated:
                raise RuntimeError("privacy controller did not activate")
            if not np.isfinite(final_bound):
                raise RuntimeError("bound cannot be computed")
            if losses[-1] > losses[0] * 2.5:
                raise RuntimeError("model diverged: loss explosion detected")

            artifact_dir.mkdir(parents=True, exist_ok=True)
            with (artifact_dir / "pac_bound.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "pac_bound": final_bound,
                        "epsilon": hp.epsilon,
                        "delta": hp.delta,
                        "n_params": n_params,
                        "retries": retries,
                    },
                    f,
                    indent=2,
                )

            with (artifact_dir / "enforcement_log.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "events": enforce_events,
                        "adjustments": adjustments,
                        "controller_activated": controller_activated,
                    },
                    f,
                    indent=2,
                )

            with (artifact_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "loss", "bound"])
                writer.writeheader()
                for i, (loss_value, bound_value) in enumerate(zip(losses, bounds)):
                    writer.writerow({"step": i, "loss": loss_value, "bound": bound_value})

            plt.figure(figsize=(6, 4))
            plt.plot(losses, label="loss")
            plt.plot(bounds, label="bound")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.title("Tiny Transformer Training Curve")
            plt.legend()
            plt.tight_layout()
            plt.savefig(artifact_dir / "loss_curve.png")
            plt.close()

            runtime = time.perf_counter() - start
            metrics = {
                "runtime_sec": runtime,
                "max_rss_mb": process.memory_info().rss / (1024 * 1024),
                "predicted_memory_mb": predicted,
                "final_hparams": hp.__dict__,
                "attempts": retries + 1,
                "numerical_issue_seen": numerical_issue,
            }
            with (artifact_dir / "runtime_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            return

        except RuntimeError as err:
            err_msg = str(err).lower()
            if (
                "out of memory" in err_msg
                or "numerical" in err_msg
                or "diverged" in err_msg
                or "no successful" in err_msg
            ):
                adjustments.append(f"retry {retries + 1}: {err}")
                hp = safer_hparams(hp)
                retries += 1
                continue
            raise

    raise RuntimeError(
        "model diverged or remained unstable after retries; check artifacts/enforcement_log.json for diagnostics"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--max-runtime-sec", type=int, default=110)
    parser.add_argument("--memory-budget-mb", type=float, default=2500.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hp = HParams()
    run_training(hp, args.max_runtime_sec, args.memory_budget_mb, args.artifact_dir)


if __name__ == "__main__":
    main()
