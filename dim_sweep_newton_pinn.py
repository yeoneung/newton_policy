#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Newton and direct PINN experiments for the chain, ring, and wide families.

The manufactured problems use periodic features and analytic spatial
derivatives. High-dimensional errors and residuals are RMS-normalized.
Use run_experiments.py for the manuscript configurations."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None


PI = math.pi
TWO_PI = 2.0 * PI
VERSION = "dim_sweep_newton_pinn_v3_dense_time_part"


# =============================================================================
# General utilities
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warning] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def np_trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device) -> dict[str, float]:
    out: dict[str, float] = {
        "cpu_rss_mb": float("nan"),
        "cuda_peak_allocated_mb": float("nan"),
        "cuda_peak_reserved_mb": float("nan"),
    }
    if psutil is not None:
        try:
            process = psutil.Process(os.getpid())
            out["cpu_rss_mb"] = process.memory_info().rss / (1024.0**2)
        except Exception:
            pass
    if device.type == "cuda":
        maybe_sync(device)
        out["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        out["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024.0**2)
    return out


class ResourceMonitor:
    """Wall-clock and peak-memory monitor for one training block."""

    def __init__(self, device: torch.device):
        self.device = device
        self.start_time = 0.0

    def __enter__(self) -> "ResourceMonitor":
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
            maybe_sync(self.device)
        self.start_time = time.perf_counter()
        return self

    def snapshot(self) -> dict[str, float]:
        snap = memory_snapshot(self.device)
        snap["seconds"] = time.perf_counter() - self.start_time
        return snap

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        pass


# =============================================================================
# Autodiff helpers
# =============================================================================


def autograd_grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Safe first derivative.  If y does not depend on x, returns zeros."""
    if not y.requires_grad:
        return torch.zeros_like(x)
    g = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros_like(x)
    return g


def laplacian_from_grad(ux: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute Delta u from ux=Du by summing coordinate second derivatives."""
    d = x.shape[1]
    lap = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for j in range(d):
        grad_ux_j = autograd_grad(ux[:, j : j + 1], x)
        lap = lap + grad_ux_j[:, j : j + 1]
    return lap


# =============================================================================
# Problem definition
# =============================================================================


@dataclass
class ProblemConfig:
    family: str  # "chain" or "ring"
    d: int
    final_time: float = 1.0
    beta_cross: float = 0.35
    gamma_cross: float = 0.25


class SweepProblem:
    """u_t + H(Du) - Delta u = S on the torus [0, 2*pi]^d."""

    def __init__(self, cfg: ProblemConfig):
        if cfg.d < 5:
            raise ValueError("This sweep requires d >= 5.")
        self.cfg = cfg
        self.family = cfg.family
        self.d = cfg.d
        self.name = f"{cfg.family}_{cfg.d}d"
        self.final_time = cfg.final_time
        self._masks: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Domain / features
    # ------------------------------------------------------------------

    def sample_x(self, *, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return TWO_PI * torch.rand(n, self.d, device=device, dtype=dtype)

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau = t / self.final_time
        return torch.cat([torch.sin(x), torch.cos(x), tau], dim=1)

    @property
    def feature_dim(self) -> int:
        return 2 * self.d + 1

    # ------------------------------------------------------------------
    # Coordinate masks (cached per device/dtype)
    # ------------------------------------------------------------------

    def _mask(self, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = f"{key}|{device}|{dtype}"
        if cache_key in self._masks:
            return self._masks[cache_key]
        d = self.d
        m = torch.zeros(d, device=device, dtype=dtype)
        half = (d + 1) // 2
        if key == "A":  # first ceil(d/2) coordinates (wide family)
            m[:half] = 1.0
        elif key == "B":  # last ceil(d/2) coordinates (wide family)
            m[d - half :] = 1.0
        elif key == "C":  # all odd coordinates (wide family)
            m[0::2] = 1.0
        elif key == "A5":  # first 5 coordinates
            m[:5] = 1.0
        elif key == "B5":  # last 5 coordinates
            m[d - 5 :] = 1.0
        elif key == "C5":  # 5 evenly spaced odd coordinates
            for idx in np.linspace(0, d - 2, 5):
                j = int(round(idx / 2.0)) * 2  # snap to even python index (odd coord)
                m[min(j, d - 1)] = 1.0
        elif key == "sgn":  # alternating signs (-1)^i for the dense time part
            m[0::2] = 1.0
            m[1::2] = -1.0
        elif key == "w":  # sparse cross direction for the Hamiltonian
            m[0] = 1.0
            m[2] = -1.0
            m[4] = 1.0
            m = m / math.sqrt(3.0)
        elif key == "even_j":  # ring family: which windows use sin
            m[0::2] = 1.0
        else:
            raise ValueError(f"Unknown mask {key}")
        self._masks[cache_key] = m
        return m

    # ------------------------------------------------------------------
    # Hamiltonian (cross-coupled nonconvex; same as the 5D/10D runs)
    # ------------------------------------------------------------------

    def H(self, p: torch.Tensor) -> torch.Tensor:
        d = p.shape[1]
        base = torch.mean(torch.sin(p) + 0.25 * torch.cos(2.0 * p), dim=1, keepdim=True)
        q_all = torch.sum(p, dim=1, keepdim=True) / math.sqrt(float(d))
        w = self._mask("w", p.device, p.dtype)
        q_sparse = torch.sum(w * p, dim=1, keepdim=True)
        return (
            base
            + self.cfg.beta_cross * torch.sin(q_all)
            + self.cfg.gamma_cross * torch.cos(q_sparse)
        )

    def Hp(self, p: torch.Tensor) -> torch.Tensor:
        d = p.shape[1]
        base = (torch.cos(p) - 0.5 * torch.sin(2.0 * p)) / float(d)
        q_all = torch.sum(p, dim=1, keepdim=True) / math.sqrt(float(d))
        all_term = self.cfg.beta_cross * torch.cos(q_all) / math.sqrt(float(d))
        w = self._mask("w", p.device, p.dtype)
        q_sparse = torch.sum(w * p, dim=1, keepdim=True)
        sparse_term = -self.cfg.gamma_cross * torch.sin(q_sparse) * w
        return base + all_term + sparse_term

    @property
    def m2_bound(self) -> float:
        return 2.0 / float(self.d) + abs(self.cfg.beta_cross) + abs(self.cfg.gamma_cross)

    # ------------------------------------------------------------------
    # Manufactured exact solutions
    # ------------------------------------------------------------------

    def exact_components(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return u, Du, Delta u, S (all closed-form)."""
        if self.family == "chain":
            return self._exact_chain(x, t)
        if self.family == "ring":
            return self._exact_ring(x, t)
        if self.family == "wide":
            return self._exact_wide(x, t)
        raise ValueError(f"Unknown family {self.family}")

    def _dense_time_part(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return P, DP, Q, DQ with Delta P = -2P and Delta Q = -2Q."""
        s = self._mask("sgn", x.device, x.dtype)
        inv = 1.0 / math.sqrt(float(self.d))
        theta = x + torch.roll(x, shifts=-1, dims=1)  # theta_i = x_i + x_{i+1}
        phi = x + torch.roll(x, shifts=-2, dims=1)  # phi_i = x_i + x_{i+2}
        P = inv * torch.sum(s * torch.cos(theta), dim=1, keepdim=True)
        Q = inv * torch.sum(s * torch.sin(phi), dim=1, keepdim=True)
        g = s * torch.sin(theta)
        DP = -inv * (g + torch.roll(g, shifts=1, dims=1))
        h = s * torch.cos(phi)
        DQ = inv * (h + torch.roll(h, shifts=2, dims=1))
        return P, DP, Q, DQ

    def _exact_chain(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse static phase + dense order-2 time part; requires d >= 10."""
        if self.d < 10:
            raise ValueError("chain family requires d >= 10")
        mA = self._mask("A5", x.device, x.dtype)
        nA = float(mA.sum())

        s1 = torch.sum(x * mA, dim=1, keepdim=True)
        P, DP, Q, DQ = self._dense_time_part(x)
        exp_t = torch.exp(-t)

        u = torch.sin(s1) + t * P + exp_t * Q
        u_t = P - exp_t * Q

        grad = torch.cos(s1) * mA + t * DP + exp_t * DQ
        lap = -nA * torch.sin(s1) - 2.0 * t * P - 2.0 * exp_t * Q

        forcing = u_t + self.H(grad) - lap
        return u, grad, lap, forcing

    def _exact_wide(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Interaction order grows with d (growing-interaction benchmark)."""
        mA = self._mask("A", x.device, x.dtype)
        mB = self._mask("B", x.device, x.dtype)
        mC = self._mask("C", x.device, x.dtype)
        nA = float(mA.sum())
        nB = float(mB.sum())
        nC = float(mC.sum())

        s1 = torch.sum(x * mA, dim=1, keepdim=True)
        s2 = torch.sum(x * mB, dim=1, keepdim=True)
        s3 = torch.sum(x * mC, dim=1, keepdim=True)
        exp_t = torch.exp(-t)

        u = torch.sin(s1) + t * torch.cos(s2) + exp_t * torch.sin(s3)
        u_t = torch.cos(s2) - exp_t * torch.sin(s3)

        grad = (
            torch.cos(s1) * mA
            - t * torch.sin(s2) * mB
            + exp_t * torch.cos(s3) * mC
        )
        lap = -nA * torch.sin(s1) - t * nB * torch.cos(s2) - exp_t * nC * torch.sin(s3)

        forcing = u_t + self.H(grad) - lap
        return u, grad, lap, forcing

    def _exact_ring(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-coordinate static coupling + dense order-2 time part; d >= 10."""
        if self.d < 10:
            raise ValueError("ring family requires d >= 10")
        even_j = self._mask("even_j", x.device, x.dtype)  # [d], 1 where f_j = sin

        # window_j = x_j + 2 x_{j+1} - x_{j+2} (cyclic)
        win = x + 2.0 * torch.roll(x, shifts=-1, dims=1) - torch.roll(x, shifts=-2, dims=1)

        f = even_j * torch.sin(win) + (1.0 - even_j) * torch.cos(win)
        fp = even_j * torch.cos(win) - (1.0 - even_j) * torch.sin(win)
        fpp = -(even_j * torch.sin(win) + (1.0 - even_j) * torch.cos(win))

        P, DP, Q, DQ = self._dense_time_part(x)
        exp_t = torch.exp(-t)

        u = torch.sum(f, dim=1, keepdim=True) + t * P + exp_t * Q
        u_t = P - exp_t * Q

        # x_i appears in win_i (coef 1), win_{i-1} (coef 2), win_{i-2} (coef -1)
        grad = (
            fp
            + 2.0 * torch.roll(fp, shifts=1, dims=1)
            - torch.roll(fp, shifts=2, dims=1)
            + t * DP
            + exp_t * DQ
        )
        # |(1, 2, -1)|^2 = 6 per window
        lap = 6.0 * torch.sum(fpp, dim=1, keepdim=True) - 2.0 * t * P - 2.0 * exp_t * Q

        forcing = u_t + self.H(grad) - lap
        return u, grad, lap, forcing

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        t0 = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return self.exact_components(x, t0)[0]

    def forcing(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.exact_components(x, t)[3]


def verify_exact(problem: SweepProblem, device: torch.device, n: int = 256) -> float:
    """Check closed-form Du/Delta u/S against float64 autograd of closed-form u.

    Computes max |R_auto[u_ex] - 0| where
        R_auto = u_t(auto) + H(Du(auto)) - Delta(auto) u - S(closed form),
    plus max component error between autograd Du and closed-form Du.
    Returns the max abs discrepancy; raises if it exceeds tolerance.
    """
    dtype = torch.float64
    x = problem.sample_x(n=n, device=device, dtype=dtype)
    t = problem.final_time * torch.rand(n, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    t.requires_grad_(True)

    u, grad_cf, lap_cf, forcing_cf = problem.exact_components(x, t)

    grad_auto = autograd_grad(u, x)
    lap_auto = laplacian_from_grad(grad_auto, x)
    ut_auto = autograd_grad(u, t)

    err_grad = float(torch.max(torch.abs(grad_auto - grad_cf)).detach().cpu())
    err_lap = float(torch.max(torch.abs(lap_auto - lap_cf)).detach().cpu())
    residual = ut_auto + problem.H(grad_auto) - lap_auto - forcing_cf
    err_res = float(torch.max(torch.abs(residual)).detach().cpu())

    err = max(err_grad, err_lap, err_res)
    print(
        f"[selfcheck] {problem.name}: max|Du err|={err_grad:.2e}, "
        f"max|Lap err|={err_lap:.2e}, max|residual|={err_res:.2e}"
    )
    if err > 1e-8:
        raise RuntimeError(
            f"Manufactured-solution self-check FAILED for {problem.name}: err={err:.3e}"
        )
    return err


# =============================================================================
# Neural models
# =============================================================================


class MLP(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NewtonPINN(nn.Module):
    """u_hat^K = g + sum_j h_j,  h_j = t * N_j(features).  IC hard-imposed."""

    def __init__(self, problem: SweepProblem):
        super().__init__()
        self.problem = problem
        self.corrections = nn.ModuleList()

    def h_from_net(self, net: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t * net(self.problem.features(x, t))

    def u(self, x: torch.Tensor, t: torch.Tensor, upto: Optional[int] = None) -> torch.Tensor:
        out = self.problem.initial_condition(x)
        nets = list(self.corrections) if upto is None else list(self.corrections)[:upto]
        for net in nets:
            out = out + self.h_from_net(net, x, t)
        return out


class DirectPINN(nn.Module):
    """Direct nonlinear PINN baseline with hard-imposed initial condition."""

    def __init__(self, problem: SweepProblem, width: int, depth: int):
        super().__init__()
        self.problem = problem
        self.net = MLP(in_dim=problem.feature_dim, width=width, depth=depth)

    def u(self, x: torch.Tensor, t: torch.Tensor, upto: Optional[int] = None) -> torch.Tensor:
        del upto
        return self.problem.initial_condition(x) + t * self.net(self.problem.features(x, t))


# =============================================================================
# Residuals
# =============================================================================


def sample_xt(
    *,
    n: int,
    problem: SweepProblem,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = problem.sample_x(n=n, device=device, dtype=dtype)
    t = problem.final_time * torch.rand(n, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_x_at_time(
    *,
    n: int,
    problem: SweepProblem,
    t_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = problem.sample_x(n=n, device=device, dtype=dtype)
    t = torch.full((n, 1), float(t_value), device=device, dtype=dtype)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def nonlinear_residual_and_grad(
    model: Any,
    x: torch.Tensor,
    t: torch.Tensor,
    upto: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """R[v] = v_t + H(Dv) - Delta v - S."""
    problem: SweepProblem = model.problem
    u_val = model.u(x, t, upto=upto)
    u_t = autograd_grad(u_val, t)
    ux = autograd_grad(u_val, x)
    lap = laplacian_from_grad(ux, x)
    residual = u_t + problem.H(ux) - lap - problem.forcing(x, t)
    return residual, ux


def correction_residual(
    *,
    model: NewtonPINN,
    net: nn.Module,
    step: int,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """rho_k = h_t + b^k . Dh - Delta h + R_k."""
    R_old, ux_old = nonlinear_residual_and_grad(model, x, t, upto=step)
    b_old = model.problem.Hp(ux_old)
    R_old = R_old.detach()
    b_old = b_old.detach()

    h = model.h_from_net(net, x, t)
    h_t = autograd_grad(h, t)
    hx = autograd_grad(h, x)
    lap_h = laplacian_from_grad(hx, x)
    return h_t + torch.sum(b_old * hx, dim=1, keepdim=True) - lap_h + R_old


def probe_residual_rms(
    *,
    model: Any,
    upto: Optional[int],
    n: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Cheap RMS estimate of the current nonlinear residual (for loss scaling)."""
    r2_sum = 0.0
    count = 0
    for start in range(0, n, batch_size):
        bsz = min(batch_size, n - start)
        x, t = sample_xt(n=bsz, problem=model.problem, device=device, dtype=dtype)
        residual, _ = nonlinear_residual_and_grad(model, x, t, upto=upto)
        r2_sum += float(torch.sum(residual.detach() ** 2).cpu())
        count += bsz
    return math.sqrt(r2_sum / max(count, 1))


# =============================================================================
# Training
# =============================================================================


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def train_newton_step(
    *,
    model: NewtonPINN,
    step: int,
    epochs: int,
    batch_size: int,
    lr: float,
    width: int,
    depth: int,
    device: torch.device,
    dtype: torch.dtype,
    log_every: int,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    freeze_module(model)
    net = MLP(in_dim=model.problem.feature_dim, width=width, depth=depth).to(
        device=device, dtype=dtype
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history: list[dict[str, Any]] = []

    # Loss preconditioning: scale by the current residual RMS (constant per step).
    sigma = max(
        probe_residual_rms(
            model=model,
            upto=step,
            n=4 * batch_size,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ),
        1e-8,
    )
    print(f"[{model.problem.name}, step={step:02d}] residual scale sigma={sigma:.4e}")

    with ResourceMonitor(device) as mon:
        final_loss = float("nan")
        for ep in range(1, epochs + 1):
            x, t = sample_xt(n=batch_size, problem=model.problem, device=device, dtype=dtype)
            rho = correction_residual(model=model, net=net, step=step, x=x, t=t)
            loss = torch.mean((rho / sigma) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.detach().cpu())
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                snap = mon.snapshot()
                history.append(
                    {
                        "problem": model.problem.name,
                        "method": "Newton-PINN",
                        "step": step,
                        "epoch": ep,
                        "loss_normalized": final_loss,
                        "sigma": sigma,
                        "seconds": snap["seconds"],
                        "cpu_rss_mb": snap["cpu_rss_mb"],
                        "cuda_peak_allocated_mb": snap["cuda_peak_allocated_mb"],
                    }
                )
                print(
                    f"[{model.problem.name}, step={step:02d}] "
                    f"epoch {ep:05d}/{epochs}, loss(norm)={final_loss:.4e}, "
                    f"time={snap['seconds']:.1f}s, peak_cuda={snap['cuda_peak_allocated_mb']:.0f}MB"
                )
        resources = mon.snapshot()
        resources["final_training_loss_normalized"] = final_loss
        resources["sigma"] = sigma

    freeze_module(net)
    return net.eval(), history, resources


def train_direct_pinn(
    *,
    model: DirectPINN,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    log_every: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, Any]] = []

    sigma = max(
        probe_residual_rms(
            model=model,
            upto=None,
            n=4 * batch_size,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        ),
        1e-8,
    )
    print(f"[{model.problem.name}, direct] residual scale sigma={sigma:.4e}")

    with ResourceMonitor(device) as mon:
        final_loss = float("nan")
        for ep in range(1, epochs + 1):
            x, t = sample_xt(n=batch_size, problem=model.problem, device=device, dtype=dtype)
            residual, _ = nonlinear_residual_and_grad(model, x, t, upto=None)
            loss = torch.mean((residual / sigma) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.detach().cpu())
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                snap = mon.snapshot()
                history.append(
                    {
                        "problem": model.problem.name,
                        "method": "Direct-PINN",
                        "step": 0,
                        "epoch": ep,
                        "loss_normalized": final_loss,
                        "sigma": sigma,
                        "seconds": snap["seconds"],
                        "cpu_rss_mb": snap["cpu_rss_mb"],
                        "cuda_peak_allocated_mb": snap["cuda_peak_allocated_mb"],
                    }
                )
                print(
                    f"[{model.problem.name}, direct] epoch {ep:05d}/{epochs}, "
                    f"loss(norm)={final_loss:.4e}, time={snap['seconds']:.1f}s, "
                    f"peak_cuda={snap['cuda_peak_allocated_mb']:.0f}MB"
                )
        resources = mon.snapshot()
        resources["final_training_loss_normalized"] = final_loss
        resources["sigma"] = sigma
    return history, resources


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_error_metrics(
    *,
    model: Any,
    problem: SweepProblem,
    upto: Optional[int],
    n_space: int,
    n_time: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """RMS-normalized Monte Carlo estimates of L2/energy errors."""
    times = torch.linspace(0.0, problem.final_time, n_time, device=device, dtype=dtype)
    l2_space_values: list[float] = []
    grad_space_values: list[float] = []

    for t_value in times:
        l2_sum = 0.0
        grad_sum = 0.0
        count = 0
        for start in range(0, n_space, batch_size):
            bsz = min(batch_size, n_space - start)
            x, t = sample_x_at_time(
                n=bsz,
                problem=problem,
                t_value=float(t_value.detach().cpu()),
                device=device,
                dtype=dtype,
            )
            u_pred = model.u(x, t, upto=upto)
            u_ex, grad_ex, _, _ = problem.exact_components(x, t)
            grad_pred = autograd_grad(u_pred, x)

            e = u_pred - u_ex
            eg = grad_pred - grad_ex
            l2_sum += float(torch.sum(e.detach() ** 2).cpu())
            grad_sum += float(torch.sum(torch.sum(eg.detach() ** 2, dim=1, keepdim=True)).cpu())
            count += bsz

        l2_space_values.append(l2_sum / count)
        grad_space_values.append(grad_sum / count)

    l2_arr = np.array(l2_space_values)
    grad_arr = np.array(grad_space_values)
    times_np = times.detach().cpu().numpy()
    l2_qt_sq = np_trapz(l2_arr, times_np) / problem.final_time
    grad_time_avg = np_trapz(grad_arr, times_np) / problem.final_time
    energy_sq = float(np.max(l2_arr) + grad_time_avg)
    final_l2_sq = float(l2_arr[-1])

    return {
        "L2_QT_RMS": math.sqrt(max(float(l2_qt_sq), 0.0)),
        "Energy_RMS": math.sqrt(max(float(energy_sq), 0.0)),
        "Final_L2x_RMS": math.sqrt(max(float(final_l2_sq), 0.0)),
        "Grad_RMS": math.sqrt(max(float(grad_time_avg), 0.0)),
    }


def estimate_residual_rms(
    *,
    model: Any,
    problem: SweepProblem,
    upto: Optional[int],
    n_mc: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    r2_sum = 0.0
    count = 0
    for start in range(0, n_mc, batch_size):
        bsz = min(batch_size, n_mc - start)
        x, t = sample_xt(n=bsz, problem=problem, device=device, dtype=dtype)
        residual, _ = nonlinear_residual_and_grad(model, x, t, upto=upto)
        r2_sum += float(torch.sum(residual.detach() ** 2).cpu())
        count += bsz
    return math.sqrt(r2_sum / count)


def estimate_certificate_rms(
    *,
    model: NewtonPINN,
    correction_index: int,
    n_mc: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """C_{k+1} = ||rho_k||_RMS + (M2/2) || |D h_k|^2 ||_RMS."""
    problem = model.problem
    net = list(model.corrections)[correction_index]
    rho2_sum = 0.0
    hx4_sum = 0.0
    count = 0

    for start in range(0, n_mc, batch_size):
        bsz = min(batch_size, n_mc - start)
        x, t = sample_xt(n=bsz, problem=problem, device=device, dtype=dtype)

        R_old, ux_old = nonlinear_residual_and_grad(model, x, t, upto=correction_index)
        b_old = problem.Hp(ux_old)
        R_old = R_old.detach()
        b_old = b_old.detach()

        h = model.h_from_net(net, x, t)
        h_t = autograd_grad(h, t)
        hx = autograd_grad(h, x)
        lap_h = laplacian_from_grad(hx, x)
        rho = h_t + torch.sum(b_old * hx, dim=1, keepdim=True) - lap_h + R_old
        hx_norm_sq = torch.sum(hx**2, dim=1, keepdim=True)

        rho2_sum += float(torch.sum(rho.detach() ** 2).cpu())
        hx4_sum += float(torch.sum((hx_norm_sq.detach()) ** 2).cpu())
        count += bsz

    rho_rms = math.sqrt(rho2_sum / count)
    quad_rms = math.sqrt(hx4_sum / count)
    certificate = rho_rms + 0.5 * problem.m2_bound * quad_rms
    return {
        "rho_RMS": rho_rms,
        "quad_RMS": quad_rms,
        "M2": problem.m2_bound,
        "Certificate_RMS": certificate,
    }


def initial_condition_check(
    *,
    model: Any,
    problem: SweepProblem,
    device: torch.device,
    dtype: torch.dtype,
    n: int = 4096,
) -> float:
    x = problem.sample_x(n=n, device=device, dtype=dtype)
    t0 = torch.zeros(n, 1, device=device, dtype=dtype)
    with torch.no_grad():
        val = model.u(x, t0, upto=None) - problem.initial_condition(x)
        return float(torch.max(torch.abs(val)).detach().cpu())


# =============================================================================
# Plotting (per-problem quicklook; combined paper figures are made separately)
# =============================================================================


def setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / f"{stem}.png", bbox_inches="tight")
    plt.close()
    print(f"[saved] {fig_dir / (stem + '.pdf')}")


def plot_problem_quicklook(fig_dir: Path, metrics: list[dict[str, Any]], problem_name: str, seed: int) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    rows = [
        r
        for r in metrics
        if r["problem"] == problem_name and r["method"] == "Newton-PINN" and r["seed"] == seed
    ]
    if not rows:
        return
    K = np.array([float(r["K"]) for r in rows])
    residual = np.array([float(r["Residual_RMS"]) for r in rows])
    cert = np.array([float(r["Certificate_RMS"]) for r in rows])
    energy = np.array([float(r["Energy_RMS"]) for r in rows])

    plt.figure(figsize=(6.6, 4.1))
    fin = np.isfinite(energy)
    plt.semilogy(K[fin], energy[fin], "o-", label="true energy error")
    plt.semilogy(K, residual, "^-", label="nonlinear residual")
    fin = np.isfinite(cert)
    plt.semilogy(K[fin], cert[fin], "s--", label="certificate")
    plt.xlabel("Newton--PINN step K")
    plt.ylabel("RMS-normalized quantity")
    plt.title(f"{problem_name} (seed {seed})")
    plt.grid(True, which="both", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    save_figure(fig_dir, f"quicklook_{problem_name}_seed{seed}")


def plot_final_slice_error(
    *,
    fig_dir: Path,
    model: NewtonPINN,
    problem: SweepProblem,
    slice_n: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()
    grid = torch.linspace(0.0, TWO_PI, slice_n, device=device, dtype=dtype)
    X1, X2 = torch.meshgrid(grid, grid, indexing="ij")
    x = torch.full((slice_n * slice_n, problem.d), 0.37 * TWO_PI, device=device, dtype=dtype)
    x[:, 0] = X1.reshape(-1)
    x[:, 1] = X2.reshape(-1)
    t = torch.full((slice_n * slice_n, 1), problem.final_time, device=device, dtype=dtype)

    with torch.no_grad():
        pred = model.u(x, t, upto=None)
        exact = problem.exact_components(x, t)[0]
        err = torch.abs(pred - exact).reshape(slice_n, slice_n).detach().cpu().numpy()

    plt.figure(figsize=(5.4, 4.4))
    plt.imshow(err, origin="lower", aspect="auto", extent=[0.0, TWO_PI, 0.0, TWO_PI])
    plt.colorbar(label="absolute error")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Final-time slice error: {problem.name}")
    save_figure(fig_dir, f"slice_error_{problem.name}_seed{seed}")


# =============================================================================
# Per-d parameter scaling
# =============================================================================


def scaled_params(d: int, args: argparse.Namespace) -> dict[str, int]:
    """Batch/eval sizes automatically reduced at high d (autograd Laplacian
    memory grows linearly with d)."""
    if args.batch_size is not None:
        train_batch = args.batch_size
    elif d <= 20:
        train_batch = 4096
    elif d <= 50:
        train_batch = 2048
    else:
        train_batch = 1024

    if args.eval_batch_size is not None:
        eval_batch = args.eval_batch_size
    elif d <= 20:
        eval_batch = 2048
    elif d <= 50:
        eval_batch = 1024
    else:
        eval_batch = 512

    if args.mc_eval is not None:
        mc_eval = args.mc_eval
    elif d <= 20:
        mc_eval = 80000
    elif d <= 50:
        mc_eval = 40000
    else:
        mc_eval = 20000

    if args.eval_space_samples is not None:
        n_space = args.eval_space_samples
    elif d <= 20:
        n_space = 8192
    elif d <= 50:
        n_space = 4096
    else:
        n_space = 2048

    return {
        "train_batch": train_batch,
        "eval_batch": eval_batch,
        "mc_eval": mc_eval,
        "n_space": n_space,
    }


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="paper", choices=["smoke", "paper", "custom"])
    parser.add_argument("--families", type=str, nargs="+", default=["chain"], choices=["chain", "ring", "wide"])
    parser.add_argument("--dims", type=int, nargs="+", default=[10, 20, 50, 100])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--outdir", type=str, default="reruns/dim_sweep")

    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--beta_cross", type=float, default=0.35)
    parser.add_argument("--gamma_cross", type=float, default=0.25)

    parser.add_argument("--newton_steps", type=int, default=None)
    parser.add_argument("--epochs_per_step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)

    parser.add_argument("--mc_eval", type=int, default=None)
    parser.add_argument("--eval_time_slices", type=int, default=None)
    parser.add_argument("--eval_space_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--slice_n", type=int, default=None)

    parser.add_argument("--with_direct", action="store_true")
    parser.add_argument("--direct_epochs", type=int, default=None)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--save_models", action="store_true")

    args = parser.parse_args()

    if args.preset == "smoke":
        defaults = {
            "newton_steps": 1,
            "epochs_per_step": 5,
            "width": 32,
            "depth": 2,
            "log_every": 1,
            "eval_time_slices": 3,
            "slice_n": 32,
            "direct_epochs": 10,
        }
        # small eval sizes for smoke
        if args.mc_eval is None:
            args.mc_eval = 512
        if args.eval_space_samples is None:
            args.eval_space_samples = 512
        if args.eval_batch_size is None:
            args.eval_batch_size = 256
        if args.batch_size is None:
            args.batch_size = 128
    else:
        defaults = {
            "newton_steps": 4,
            "epochs_per_step": 4000,
            "width": 128,
            "depth": 5,
            "log_every": 500,
            "eval_time_slices": 31,
            "slice_n": 160,
            "direct_epochs": 16000,
        }

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def metrics_row(
    *,
    problem: SweepProblem,
    seed: int,
    method: str,
    K: int,
    err: dict[str, float],
    residual_rms: float,
    cert: Optional[dict[str, float]],
    ic_err: float,
    train_seconds: float,
    cum_train_seconds: float,
    peak_cuda_mb: float,
) -> dict[str, Any]:
    certificate = cert["Certificate_RMS"] if cert is not None else float("nan")
    energy = err.get("Energy_RMS", float("nan"))
    return {
        "problem": problem.name,
        "family": problem.family,
        "d": problem.d,
        "seed": seed,
        "method": method,
        "K": K,
        "L2_QT_RMS": err.get("L2_QT_RMS", float("nan")),
        "Energy_RMS": energy,
        "Final_L2x_RMS": err.get("Final_L2x_RMS", float("nan")),
        "Grad_RMS": err.get("Grad_RMS", float("nan")),
        "Residual_RMS": residual_rms,
        "rho_RMS": cert["rho_RMS"] if cert is not None else float("nan"),
        "quad_RMS": cert["quad_RMS"] if cert is not None else float("nan"),
        "M2": problem.m2_bound,
        "Certificate_RMS": certificate,
        "Energy_over_Cert": energy / certificate
        if np.isfinite(energy) and np.isfinite(certificate) and certificate > 0
        else float("nan"),
        "IC_max_error": ic_err,
        "train_seconds": train_seconds,
        "cum_train_seconds": cum_train_seconds,
        "cuda_peak_allocated_mb": peak_cuda_mb,
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    outdir = Path(args.outdir)
    table_dir = outdir / "tables"
    fig_dir = outdir / "figures"
    model_dir = outdir / "models"
    for p in (table_dir, fig_dir, model_dir):
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Dimension-sweep Newton--PINN benchmark")
    print(f"code version : {VERSION}")
    print(f"time         : {now_string()}")
    print(f"device       : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")
    print(f"torch        : {torch.__version__}")
    print(f"families     : {args.families}")
    print(f"dims         : {args.dims}")
    print(f"seeds        : {args.seeds}")
    print(f"newton_steps : {args.newton_steps}, epochs/step {args.epochs_per_step}")
    print(f"with_direct  : {args.with_direct} ({args.direct_epochs} epochs)")
    print(f"output       : {outdir.resolve()}")
    print("=" * 80)

    all_metrics: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    all_resources: list[dict[str, Any]] = []

    t_run0 = time.time()

    for family in args.families:
        for d in args.dims:
            cfg = ProblemConfig(
                family=family,
                d=d,
                final_time=args.T,
                beta_cross=args.beta_cross,
                gamma_cross=args.gamma_cross,
            )
            problem = SweepProblem(cfg)
            verify_exact(problem, device)
            sp = scaled_params(d, args)
            print(
                f"\n=== {problem.name}: train_batch={sp['train_batch']}, "
                f"eval_batch={sp['eval_batch']}, mc_eval={sp['mc_eval']}, "
                f"M2={problem.m2_bound:.3f} ==="
            )

            for seed in args.seeds:
                set_seed(seed)
                print(f"\n--- {problem.name}, seed={seed} ---")

                model = NewtonPINN(problem).to(device=device, dtype=dtype)
                ic0 = initial_condition_check(model=model, problem=problem, device=device, dtype=dtype)

                # K=0 metrics
                m0 = evaluate_error_metrics(
                    model=model,
                    problem=problem,
                    upto=0,
                    n_space=sp["n_space"],
                    n_time=args.eval_time_slices,
                    batch_size=sp["eval_batch"],
                    device=device,
                    dtype=dtype,
                )
                r0 = estimate_residual_rms(
                    model=model,
                    problem=problem,
                    upto=0,
                    n_mc=sp["mc_eval"],
                    batch_size=sp["eval_batch"],
                    device=device,
                    dtype=dtype,
                )
                all_metrics.append(
                    metrics_row(
                        problem=problem,
                        seed=seed,
                        method="Newton-PINN",
                        K=0,
                        err=m0,
                        residual_rms=r0,
                        cert=None,
                        ic_err=ic0,
                        train_seconds=float("nan"),
                        cum_train_seconds=0.0,
                        peak_cuda_mb=float("nan"),
                    )
                )

                cum_seconds = 0.0
                for k in range(args.newton_steps):
                    net, hist, res = train_newton_step(
                        model=model,
                        step=k,
                        epochs=args.epochs_per_step,
                        batch_size=sp["train_batch"],
                        lr=args.lr,
                        width=args.width,
                        depth=args.depth,
                        device=device,
                        dtype=dtype,
                        log_every=args.log_every,
                    )
                    model.corrections.append(net)
                    for h in hist:
                        h["seed"] = seed
                        h["family"] = family
                        h["d"] = d
                    all_history.extend(hist)
                    cum_seconds += res["seconds"]
                    all_resources.append(
                        {
                            "problem": problem.name,
                            "family": family,
                            "d": d,
                            "seed": seed,
                            "method": "Newton-PINN",
                            "step": k,
                            "train_seconds": res["seconds"],
                            "cuda_peak_allocated_mb": res["cuda_peak_allocated_mb"],
                            "cuda_peak_reserved_mb": res["cuda_peak_reserved_mb"],
                            "cpu_rss_mb": res["cpu_rss_mb"],
                            "sigma": res["sigma"],
                        }
                    )

                    mK = evaluate_error_metrics(
                        model=model,
                        problem=problem,
                        upto=k + 1,
                        n_space=sp["n_space"],
                        n_time=args.eval_time_slices,
                        batch_size=sp["eval_batch"],
                        device=device,
                        dtype=dtype,
                    )
                    rK = estimate_residual_rms(
                        model=model,
                        problem=problem,
                        upto=k + 1,
                        n_mc=sp["mc_eval"],
                        batch_size=sp["eval_batch"],
                        device=device,
                        dtype=dtype,
                    )
                    cert = estimate_certificate_rms(
                        model=model,
                        correction_index=k,
                        n_mc=sp["mc_eval"],
                        batch_size=sp["eval_batch"],
                        device=device,
                        dtype=dtype,
                    )
                    ic_err = initial_condition_check(
                        model=model, problem=problem, device=device, dtype=dtype
                    )
                    all_metrics.append(
                        metrics_row(
                            problem=problem,
                            seed=seed,
                            method="Newton-PINN",
                            K=k + 1,
                            err=mK,
                            residual_rms=rK,
                            cert=cert,
                            ic_err=ic_err,
                            train_seconds=res["seconds"],
                            cum_train_seconds=cum_seconds,
                            peak_cuda_mb=res["cuda_peak_allocated_mb"],
                        )
                    )
                    print(
                        f"[{problem.name}, seed={seed}, K={k + 1}] "
                        f"E={mK['Energy_RMS']:.3e}, R={rK:.3e}, "
                        f"C={cert['Certificate_RMS']:.3e}, "
                        f"ratio={mK['Energy_RMS'] / cert['Certificate_RMS']:.3f}"
                    )

                if args.save_models:
                    torch.save(
                        {"cfg": cfg, "state_dict": model.state_dict(), "version": VERSION},
                        model_dir / f"newton_{problem.name}_seed{seed}.pt",
                    )

                if not args.skip_plots:
                    plot_problem_quicklook(fig_dir, all_metrics, problem.name, seed)
                    plot_final_slice_error(
                        fig_dir=fig_dir,
                        model=model,
                        problem=problem,
                        slice_n=args.slice_n,
                        device=device,
                        dtype=dtype,
                        seed=seed,
                    )

                # ----------------------------------------------------------
                # Direct PINN baseline
                # ----------------------------------------------------------
                if args.with_direct:
                    set_seed(seed + 1000)
                    direct = DirectPINN(problem, width=args.width, depth=args.depth).to(
                        device=device, dtype=dtype
                    )
                    dhist, dres = train_direct_pinn(
                        model=direct,
                        epochs=args.direct_epochs,
                        batch_size=sp["train_batch"],
                        lr=args.lr,
                        device=device,
                        dtype=dtype,
                        log_every=args.log_every,
                    )
                    for h in dhist:
                        h["seed"] = seed
                        h["family"] = family
                        h["d"] = d
                    all_history.extend(dhist)
                    all_resources.append(
                        {
                            "problem": problem.name,
                            "family": family,
                            "d": d,
                            "seed": seed,
                            "method": "Direct-PINN",
                            "step": 0,
                            "train_seconds": dres["seconds"],
                            "cuda_peak_allocated_mb": dres["cuda_peak_allocated_mb"],
                            "cuda_peak_reserved_mb": dres["cuda_peak_reserved_mb"],
                            "cpu_rss_mb": dres["cpu_rss_mb"],
                            "sigma": dres["sigma"],
                        }
                    )

                    md = evaluate_error_metrics(
                        model=direct,
                        problem=problem,
                        upto=None,
                        n_space=sp["n_space"],
                        n_time=args.eval_time_slices,
                        batch_size=sp["eval_batch"],
                        device=device,
                        dtype=dtype,
                    )
                    rd = estimate_residual_rms(
                        model=direct,
                        problem=problem,
                        upto=None,
                        n_mc=sp["mc_eval"],
                        batch_size=sp["eval_batch"],
                        device=device,
                        dtype=dtype,
                    )
                    icd = initial_condition_check(
                        model=direct, problem=problem, device=device, dtype=dtype
                    )
                    all_metrics.append(
                        metrics_row(
                            problem=problem,
                            seed=seed,
                            method="Direct-PINN",
                            K=args.direct_epochs,
                            err=md,
                            residual_rms=rd,
                            cert=None,
                            ic_err=icd,
                            train_seconds=dres["seconds"],
                            cum_train_seconds=dres["seconds"],
                            peak_cuda_mb=dres["cuda_peak_allocated_mb"],
                        )
                    )
                    print(
                        f"[{problem.name}, seed={seed}, direct] "
                        f"E={md['Energy_RMS']:.3e}, R={rd:.3e}, time={dres['seconds']:.0f}s"
                    )
                    if args.save_models:
                        torch.save(
                            {"cfg": cfg, "state_dict": direct.state_dict(), "version": VERSION},
                            model_dir / f"direct_{problem.name}_seed{seed}.pt",
                        )

                # Incremental persistence after every (problem, seed) block
                write_csv(table_dir / "metrics.csv", all_metrics)
                write_csv(table_dir / "training_history.csv", all_history)
                write_csv(table_dir / "iteration_resources.csv", all_resources)

    elapsed = time.time() - t_run0
    with (outdir / "run_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Dimension-sweep Newton--PINN benchmark\n")
        f.write(f"version: {VERSION}\n")
        f.write(f"time: {now_string()}\n")
        f.write(f"device: {device}\n")
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"families: {args.families}\n")
        f.write(f"dims: {args.dims}\n")
        f.write(f"seeds: {args.seeds}\n")
        f.write(f"newton_steps: {args.newton_steps}\n")
        f.write(f"epochs_per_step: {args.epochs_per_step}\n")
        f.write(f"direct_epochs: {args.direct_epochs} (with_direct={args.with_direct})\n")
        f.write(f"width/depth: {args.width}/{args.depth}\n")
        f.write(f"beta_cross/gamma_cross: {args.beta_cross}/{args.gamma_cross}\n")
        f.write(f"total wall clock: {elapsed:.1f} s\n")
        f.write("Norms are RMS-normalized Monte Carlo estimates.\n")
        f.write("Training loss is preconditioned by the per-step residual RMS (reported values are unnormalized).\n")

    print(f"\nDone in {elapsed / 60.0:.1f} min.")
    print(f"Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
