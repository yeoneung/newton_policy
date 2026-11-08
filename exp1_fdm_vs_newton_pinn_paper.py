#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-dimensional Cole--Hopf benchmark with a finite-difference reference.

Newton corrections satisfy the initial condition exactly. Training uses
Adam followed by optional L-BFGS refinement."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Global constants
# ============================================================

PI = math.pi
DOMAIN_LENGTH = 2.0 * PI
VERSION = "v3_exact_initial_residual_safe_autograd"


# ============================================================
# Reproducibility and device
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warning] CUDA requested but torch.cuda.is_available() is False. Using CPU.")
        return torch.device("cpu")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def now_string() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Exact solution and PDE residual
# ============================================================

def exact_phi(x: torch.Tensor, t: torch.Tensor, a: float) -> torch.Tensor:
    return 1.0 + a * torch.exp(-t) * torch.cos(x)


def exact_u(x: torch.Tensor, t: torch.Tensor, a: float) -> torch.Tensor:
    return -2.0 * torch.log(exact_phi(x, t, a))


def exact_g(x: torch.Tensor, a: float) -> torch.Tensor:
    return -2.0 * torch.log(1.0 + a * torch.cos(x))


def autograd_grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Safe autograd derivative.

    When K=0, u_hat^0(x,t)=g(x) has no t-dependence, so u_t should be zero.
    PyTorch then reports that t was not used in the graph.  In that case,
    we return the mathematically correct zero derivative.
    """
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


def nonlinear_residual_and_ux(
    u: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    R[u] = u_t + 0.5 u_x^2 - u_xx.

    For H(p)=0.5p^2:
        H_p(p)=p,
        H_pp(p)=1.
    """
    u_t = autograd_grad(u, t)
    u_x = autograd_grad(u, x)
    u_xx = autograd_grad(u_x, x)
    residual = u_t + 0.5 * u_x**2 - u_xx
    return residual, u_x


def initial_residual_and_ux(
    x: torch.Tensor,
    a: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Residual of the initial guess u_hat^0(x,t)=g(x).

    This function intentionally does NOT differentiate with respect to t.
    Since u_hat^0 has no t-dependence, u_t=0 exactly.  This avoids the
    PyTorch unused-tensor issue at K=0.
    """
    g = exact_g(x, a)
    g_x = autograd_grad(g, x)
    g_xx = autograd_grad(g_x, x)
    residual = 0.5 * g_x**2 - g_xx
    return residual, g_x


# ============================================================
# Finite difference baseline
# ============================================================

def periodic_dx(u: torch.Tensor, dx: float) -> torch.Tensor:
    return (torch.roll(u, shifts=-1, dims=-1) - torch.roll(u, shifts=1, dims=-1)) / (2.0 * dx)


def fdm_semimplicit(
    *,
    nx: int,
    nt: int,
    final_time: float,
    a: float,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Semi-implicit periodic FDM:

        (u^{n+1}-u^n)/dt - Dxx u^{n+1}
            = -0.5 (Dx u^n)^2.

    The periodic cyclic system is diagonalized by FFT.
    """
    dx = DOMAIN_LENGTH / nx
    dt = final_time / nt

    x = torch.arange(nx, device=device, dtype=dtype) * dx
    t_grid = torch.linspace(0.0, final_time, nt + 1, device=device, dtype=dtype)

    u = exact_g(x.reshape(-1, 1), a).reshape(nx)
    sol = torch.empty((nt + 1, nx), device=device, dtype=dtype)
    sol[0] = u

    modes = torch.fft.fftfreq(nx, d=1.0 / nx, device=device).to(dtype=dtype)
    lap_symbol = -4.0 * torch.sin(PI * modes / nx) ** 2 / (dx * dx)
    denom = 1.0 - dt * lap_symbol

    for n in range(nt):
        ux = periodic_dx(u, dx)
        rhs = u - dt * 0.5 * ux**2
        rhs_hat = torch.fft.fft(rhs)
        u = torch.fft.ifft(rhs_hat / denom).real
        sol[n + 1] = u

    return x, t_grid, sol


# ============================================================
# Neural model
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int = 3, width: int = 96, depth: int = 5):
        super().__init__()
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
    """
    Newton--PINN iterate:

        u_hat^K(x,t) = g(x) + sum_{j=0}^{K-1} h_j(x,t),
        h_j(x,t) = t N_j(sin x, cos x, t/T).

    Periodicity is encoded through sin/cos features.
    Initial trace is exact because h_j(x,0)=0.
    """
    def __init__(self, *, a: float, final_time: float):
        super().__init__()
        self.a = float(a)
        self.final_time = float(final_time)
        self.corrections = nn.ModuleList()

    def features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau = t / self.final_time
        return torch.cat([torch.sin(x), torch.cos(x), tau], dim=1)

    def h_from_net(self, net: nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return t * net(self.features(x, t))

    def u(self, x: torch.Tensor, t: torch.Tensor, upto: int | None = None) -> torch.Tensor:
        out = exact_g(x, self.a)
        nets = list(self.corrections) if upto is None else list(self.corrections)[:upto]
        for net in nets:
            out = out + self.h_from_net(net, x, t)
        return out


# ============================================================
# Training
# ============================================================

@dataclass
class StepRecord:
    step: int
    adam_final_loss: float
    lbfgs_final_loss: float | None
    train_seconds: float


def sample_collocation(
    *,
    n: int,
    final_time: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = DOMAIN_LENGTH * torch.rand(n, 1, device=device, dtype=dtype)
    t = final_time * torch.rand(n, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def correction_residual(
    *,
    model: NewtonPINN,
    net: nn.Module,
    step: int,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    rho_k = h_t + b^k h_x - h_xx + R_k,
    where
        R_k = R[u_hat^k],
        b^k = H_p((u_hat^k)_x) = (u_hat^k)_x.
    """
    if step == 0:
        R_old, ux_old = initial_residual_and_ux(x, model.a)
    else:
        u_old = model.u(x, t, upto=step)
        R_old, ux_old = nonlinear_residual_and_ux(u_old, x, t)

    # Freeze the Newton linearization target during correction training.
    R_old = R_old.detach()
    b_old = ux_old.detach()

    h = model.h_from_net(net, x, t)
    h_t = autograd_grad(h, t)
    h_x = autograd_grad(h, x)
    h_xx = autograd_grad(h_x, x)

    return h_t + b_old * h_x - h_xx + R_old


def train_newton_step(
    *,
    model: NewtonPINN,
    step: int,
    epochs: int,
    batch_size: int,
    lr: float,
    width: int,
    depth: int,
    lbfgs_iters: int,
    lbfgs_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    print_every: int,
) -> StepRecord:
    t0 = time.time()

    for p in model.parameters():
        p.requires_grad_(False)

    net = MLP(in_dim=3, width=width, depth=depth).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    adam_final_loss = float("nan")
    for ep in range(1, epochs + 1):
        x, t = sample_collocation(
            n=batch_size,
            final_time=model.final_time,
            device=device,
            dtype=dtype,
        )

        rho = correction_residual(model=model, net=net, step=step, x=x, t=t)
        loss = torch.mean(rho**2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        adam_final_loss = float(loss.detach().cpu())

        if print_every > 0 and (ep == 1 or ep % print_every == 0 or ep == epochs):
            print(f"[step {step:02d}] Adam {ep:05d}/{epochs}, loss={adam_final_loss:.4e}")

    lbfgs_final_loss: float | None = None
    if lbfgs_iters > 0:
        print(f"[step {step:02d}] Starting LBFGS refinement with {lbfgs_iters} iterations.")
        opt_lbfgs = torch.optim.LBFGS(
            net.parameters(),
            lr=1.0,
            max_iter=lbfgs_iters,
            max_eval=lbfgs_iters * 2,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        x_fixed0 = DOMAIN_LENGTH * torch.rand(lbfgs_batch_size, 1, device=device, dtype=dtype)
        t_fixed0 = model.final_time * torch.rand(lbfgs_batch_size, 1, device=device, dtype=dtype)

        def closure() -> torch.Tensor:
            opt_lbfgs.zero_grad(set_to_none=True)
            x_fixed = x_fixed0.clone().detach().requires_grad_(True)
            t_fixed = t_fixed0.clone().detach().requires_grad_(True)
            rho_fixed = correction_residual(model=model, net=net, step=step, x=x_fixed, t=t_fixed)
            loss_fixed = torch.mean(rho_fixed**2)
            loss_fixed.backward()
            return loss_fixed

        loss_after = opt_lbfgs.step(closure)
        lbfgs_final_loss = float(loss_after.detach().cpu())
        print(f"[step {step:02d}] LBFGS done, loss={lbfgs_final_loss:.4e}")

    for p in net.parameters():
        p.requires_grad_(False)
    model.corrections.append(net.eval())

    return StepRecord(
        step=step,
        adam_final_loss=adam_final_loss,
        lbfgs_final_loss=lbfgs_final_loss,
        train_seconds=time.time() - t0,
    )


# ============================================================
# Metrics
# ============================================================

def discrete_error_metrics(
    *,
    u_num: torch.Tensor,
    x: torch.Tensor,
    t_grid: torch.Tensor,
    a: float,
) -> dict[str, float]:
    """
    Computes physical unnormalized norms:

        L2_QT^2 = int_0^T int_T |e|^2 dx dt,

        Energy^2 = sup_t int_T |e|^2 dx
                   + int_0^T int_T |e_x|^2 dx dt.

    The gradient part is evaluated by periodic central differences on the
    evaluation grid.
    """
    nx = x.numel()
    nt = t_grid.numel() - 1
    dx = DOMAIN_LENGTH / nx
    dt = float((t_grid[-1] - t_grid[0]).detach().cpu()) / nt

    X = x.reshape(1, nx)
    TT = t_grid.reshape(nt + 1, 1)
    u_ex = exact_u(X, TT, a).to(device=u_num.device, dtype=u_num.dtype)

    e = u_num - u_ex
    e_x = periodic_dx(e, dx)

    trap_w = torch.ones(nt + 1, device=u_num.device, dtype=u_num.dtype)
    trap_w[0] = 0.5
    trap_w[-1] = 0.5

    l2_qt = torch.sqrt(torch.sum(trap_w.reshape(-1, 1) * e**2) * dx * dt)
    sup_l2x_sq = torch.max(torch.sum(e**2, dim=1) * dx)
    grad_int = torch.sum(trap_w.reshape(-1, 1) * e_x**2) * dx * dt
    energy = torch.sqrt(sup_l2x_sq + grad_int)
    final_l2x = torch.sqrt(torch.sum(e[-1] ** 2) * dx)

    return {
        "L2_QT": float(l2_qt.detach().cpu()),
        "Energy": float(energy.detach().cpu()),
        "Final_L2x": float(final_l2x.detach().cpu()),
    }


@torch.no_grad()
def eval_pinn_on_grid(
    *,
    model: NewtonPINN,
    x: torch.Tensor,
    t_grid: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    upto: int | None = None,
    batch: int = 65536,
) -> torch.Tensor:
    """
    Evaluates u_hat on a tensor-product grid.
    Output shape: [Nt+1, Nx].
    """
    nx = x.numel()
    ntp1 = t_grid.numel()

    TT, XX = torch.meshgrid(t_grid.to(device=device, dtype=dtype), x.to(device=device, dtype=dtype), indexing="ij")
    xf = XX.reshape(-1, 1)
    tf = TT.reshape(-1, 1)

    chunks = []
    for i in range(0, xf.shape[0], batch):
        chunks.append(model.u(xf[i:i + batch], tf[i:i + batch], upto=upto).detach())

    return torch.cat(chunks, dim=0).reshape(ntp1, nx).contiguous()


def estimate_certificate_for_correction(
    *,
    model: NewtonPINN,
    correction_index: int,
    n_mc: int,
    device: torch.device,
    dtype: torch.dtype,
    batch: int = 8192,
) -> dict[str, float]:
    """
    For correction h^k, estimate

        C_{k+1} = ||rho_k||_{L2(QT)}
                  + 0.5 || |h_x^k|^2 ||_{L2(QT)}.

    Since H(p)=0.5p^2, M2=1.
    """
    assert 0 <= correction_index < len(model.corrections)

    volume = DOMAIN_LENGTH * model.final_time
    rho2_sum = 0.0
    hx4_sum = 0.0
    count = 0

    net = list(model.corrections)[correction_index]

    for start in range(0, n_mc, batch):
        bsz = min(batch, n_mc - start)
        x, t = sample_collocation(
            n=bsz,
            final_time=model.final_time,
            device=device,
            dtype=dtype,
        )

        if correction_index == 0:
            R_old, ux_old = initial_residual_and_ux(x, model.a)
        else:
            u_old = model.u(x, t, upto=correction_index)
            R_old, ux_old = nonlinear_residual_and_ux(u_old, x, t)

        R_old = R_old.detach()
        b_old = ux_old.detach()

        h = model.h_from_net(net, x, t)
        h_t = autograd_grad(h, t)
        h_x = autograd_grad(h, x)
        h_xx = autograd_grad(h_x, x)

        rho = h_t + b_old * h_x - h_xx + R_old

        rho2_sum += float(torch.sum(rho.detach() ** 2).cpu())
        hx4_sum += float(torch.sum(h_x.detach() ** 4).cpu())
        count += bsz

    rho_l2 = math.sqrt(volume * rho2_sum / count)
    hx_sq_l2 = math.sqrt(volume * hx4_sum / count)
    certificate = rho_l2 + 0.5 * hx_sq_l2

    return {
        "rho_l2": rho_l2,
        "hx_sq_l2": hx_sq_l2,
        "certificate": certificate,
    }


def estimate_final_nonlinear_residual(
    *,
    model: NewtonPINN,
    upto: int | None,
    n_mc: int,
    device: torch.device,
    dtype: torch.dtype,
    batch: int = 8192,
) -> float:
    volume = DOMAIN_LENGTH * model.final_time
    r2_sum = 0.0
    count = 0

    for start in range(0, n_mc, batch):
        bsz = min(batch, n_mc - start)
        x, t = sample_collocation(
            n=bsz,
            final_time=model.final_time,
            device=device,
            dtype=dtype,
        )
        if upto == 0:
            residual, _ = initial_residual_and_ux(x, model.a)
        else:
            u_val = model.u(x, t, upto=upto)
            residual, _ = nonlinear_residual_and_ux(u_val, x, t)

        r2_sum += float(torch.sum(residual.detach() ** 2).cpu())
        count += bsz

    return math.sqrt(volume * r2_sum / count)


# ============================================================
# CSV and table helpers
# ============================================================

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_float(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.4e}"
    return str(x)


def print_metrics_table(rows: list[dict[str, Any]]) -> None:
    columns = ["method", "K", "L2_QT", "Energy", "Final_L2x", "Residual_L2", "Certificate", "Energy_over_Cert"]
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(format_float(row.get(c, ""))))

    print("")
    print(" | ".join(c.ljust(widths[c]) for c in columns))
    print("-+-".join("-" * widths[c] for c in columns))
    for row in rows:
        print(" | ".join(format_float(row.get(c, "")).ljust(widths[c]) for c in columns))
    print("")


# ============================================================
# Plotting
# ============================================================

def setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_current_figure(fig_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fig_dir / f"{stem}.pdf"
    png_path = fig_dir / f"{stem}.png"
    plt.tight_layout()
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"[saved] {pdf_path}")
    print(f"[saved] {png_path}")


def plot_final_profile(
    *,
    fig_dir: Path,
    x_eval: torch.Tensor,
    t_eval: torch.Tensor,
    u_pinn: torch.Tensor,
    x_fdm: torch.Tensor,
    t_fdm: torch.Tensor,
    u_fdm: torch.Tensor,
    a: float,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()

    x_np = x_eval.detach().cpu().numpy()
    tT = torch.full_like(x_eval, float(t_eval[-1].detach().cpu()))
    exact_T = exact_u(x_eval, tT, a).detach().cpu().numpy()

    x_fdm_np = x_fdm.detach().cpu().numpy()
    fdm_T = u_fdm[-1].detach().cpu().numpy()
    pinn_T = u_pinn[-1].detach().cpu().numpy()

    plt.figure(figsize=(6.3, 4.0))
    plt.plot(x_np, exact_T, label="Exact")
    plt.plot(x_fdm_np, fdm_T, "--", label="FDM")
    plt.plot(x_np, pinn_T, ":", linewidth=2.0, label="Newton--PINN")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$u(x,T)$")
    plt.title("Final-time solution profile")
    plt.legend(frameon=False)
    save_current_figure(fig_dir, "fig01_final_time_profile")


def plot_error_vs_step(
    *,
    fig_dir: Path,
    metrics_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()

    rows = [r for r in metrics_rows if r["method"] == "Newton-PINN"]
    K = np.array([r["K"] for r in rows], dtype=float)
    energy = np.array([r["Energy"] for r in rows], dtype=float)
    cert = np.array([r["Certificate"] for r in rows], dtype=float)

    plt.figure(figsize=(6.3, 4.0))
    plt.semilogy(K, energy, "o-", label=r"true error $E_K$")
    valid = np.isfinite(cert)
    plt.semilogy(K[valid], cert[valid], "s--", label=r"certificate $\mathfrak{C}_K$")
    plt.xlabel("Newton--PINN step K")
    plt.ylabel("quantity")
    plt.title("Error and residual certificate")
    plt.legend(frameon=False)
    plt.grid(True, which="both", linewidth=0.5, alpha=0.5)
    save_current_figure(fig_dir, "fig02_error_and_certificate")


def plot_certificate_ratio(
    *,
    fig_dir: Path,
    metrics_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()

    rows = [r for r in metrics_rows if r["method"] == "Newton-PINN" and r["K"] >= 1]
    K = np.array([r["K"] for r in rows], dtype=float)
    ratio = np.array([r["Energy_over_Cert"] for r in rows], dtype=float)

    plt.figure(figsize=(6.3, 4.0))
    plt.plot(K, ratio, "o-")
    plt.xlabel("Newton--PINN step K")
    plt.ylabel(r"$E_K/\mathfrak{C}_K$")
    plt.title("Certificate calibration ratio")
    plt.grid(True, linewidth=0.5, alpha=0.5)
    save_current_figure(fig_dir, "fig03_certificate_ratio")


def plot_training_loss(
    *,
    fig_dir: Path,
    loss_history: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()

    plt.figure(figsize=(6.3, 4.0))
    for step in sorted(set(int(r["step"]) for r in loss_history)):
        step_rows = [r for r in loss_history if int(r["step"]) == step]
        epoch = np.array([r["epoch"] for r in step_rows], dtype=float)
        loss = np.array([r["loss"] for r in step_rows], dtype=float)
        plt.semilogy(epoch, loss, label=f"step {step}")
    plt.xlabel("Adam epoch")
    plt.ylabel("training loss")
    plt.title("Linearized residual training loss")
    plt.legend(frameon=False)
    plt.grid(True, which="both", linewidth=0.5, alpha=0.5)
    save_current_figure(fig_dir, "fig04_training_loss")


def plot_spacetime_error(
    *,
    fig_dir: Path,
    x_eval: torch.Tensor,
    t_eval: torch.Tensor,
    u_pinn: torch.Tensor,
    a: float,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()

    nx = x_eval.numel()
    ntp1 = t_eval.numel()
    X = x_eval.reshape(1, nx)
    TT = t_eval.reshape(ntp1, 1)
    u_ex = exact_u(X, TT, a).to(device=u_pinn.device, dtype=u_pinn.dtype)
    err = torch.abs(u_pinn - u_ex).detach().cpu().numpy()

    plt.figure(figsize=(6.3, 4.2))
    plt.imshow(
        err,
        origin="lower",
        aspect="auto",
        extent=[
            float(x_eval[0].detach().cpu()),
            float(x_eval[-1].detach().cpu()),
            float(t_eval[0].detach().cpu()),
            float(t_eval[-1].detach().cpu()),
        ],
    )
    plt.colorbar(label=r"$|\widehat u^K-u_{\rm ex}|$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$t$")
    plt.title("Space-time absolute error")
    save_current_figure(fig_dir, "fig05_spacetime_error")


def plot_final_error_profile(
    *,
    fig_dir: Path,
    x_eval: torch.Tensor,
    t_eval: torch.Tensor,
    u_pinn: torch.Tensor,
    a: float,
) -> None:
    import matplotlib.pyplot as plt

    setup_matplotlib()

    tT = torch.full_like(x_eval, float(t_eval[-1].detach().cpu()))
    exact_T = exact_u(x_eval, tT, a).to(device=u_pinn.device, dtype=u_pinn.dtype)
    err_T = torch.abs(u_pinn[-1] - exact_T).detach().cpu().numpy()

    plt.figure(figsize=(6.3, 4.0))
    plt.semilogy(x_eval.detach().cpu().numpy(), err_T)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$|\widehat u^K(x,T)-u_{\rm ex}(x,T)|$")
    plt.title("Final-time absolute error")
    plt.grid(True, which="both", linewidth=0.5, alpha=0.5)
    save_current_figure(fig_dir, "fig06_final_time_error")


# ============================================================
# Presets and argument parsing
# ============================================================

def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Presets are applied only to values that were not explicitly overwritten by
    the user.  We keep this simple by setting preset defaults before parsing in
    parse_args().
    """
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--preset", type=str, default="quick", choices=["quick", "paper", "custom"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="reruns/exp1_convergence")

    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=1.0)

    # Numerical evaluation grid
    parser.add_argument("--eval_nx", type=int, default=None)
    parser.add_argument("--eval_nt", type=int, default=None)

    # FDM grid
    parser.add_argument("--fdm_nx", type=int, default=None)
    parser.add_argument("--fdm_nt", type=int, default=None)

    # Newton--PINN
    parser.add_argument("--newton_steps", type=int, default=None)
    parser.add_argument("--epochs_per_step", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--lbfgs_iters", type=int, default=None)
    parser.add_argument("--lbfgs_batch_size", type=int, default=None)

    # Evaluation
    parser.add_argument("--mc_eval", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float64"])

    args = parser.parse_args()

    if args.preset == "quick":
        defaults = {
            "eval_nx": 256,
            "eval_nt": 160,
            "fdm_nx": 256,
            "fdm_nt": 1600,
            "newton_steps": 3,
            "epochs_per_step": 800,
            "batch_size": 2048,
            "width": 64,
            "depth": 4,
            "lbfgs_iters": 0,
            "lbfgs_batch_size": 4096,
            "mc_eval": 20000,
            "print_every": 200,
        }
    elif args.preset == "paper":
        defaults = {
            "eval_nx": 512,
            "eval_nt": 240,
            "fdm_nx": 512,
            "fdm_nt": 4000,
            "newton_steps": 5,
            "epochs_per_step": 4000,
            "batch_size": 4096,
            "width": 96,
            "depth": 5,
            "lbfgs_iters": 400,
            "lbfgs_batch_size": 8192,
            "mc_eval": 80000,
            "print_every": 500,
        }
    else:
        defaults = {
            "eval_nx": 256,
            "eval_nt": 160,
            "fdm_nx": 256,
            "fdm_nt": 1600,
            "newton_steps": 4,
            "epochs_per_step": 1500,
            "batch_size": 4096,
            "width": 64,
            "depth": 4,
            "lbfgs_iters": 0,
            "lbfgs_batch_size": 4096,
            "mc_eval": 30000,
            "print_every": 500,
        }

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.dtype is None:
        args.dtype = "float64" if args.preset == "paper" else "float32"
    return args


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    args = parse_args()

    if not (0.0 < args.a < 1.0):
        raise ValueError("Need 0 < a < 1 so that phi=1+a exp(-t) cos x stays positive.")

    set_seed(args.seed)
    device = get_device(args.device)
    pinn_dtype = torch.float32 if args.dtype == "float32" else torch.float64

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    outdir = Path(args.outdir)
    fig_dir = outdir / "figures"
    table_dir = outdir / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Experiment 1: FDM vs Newton--PINN")
    print(f"code version          : {VERSION}")
    print("=" * 72)
    print(f"time                 : {now_string()}")
    print(f"device               : {device}")
    print(f"torch version         : {torch.__version__}")
    print(f"preset               : {args.preset}")
    print(f"output directory      : {outdir.resolve()}")
    print(f"a                    : {args.a}")
    print(f"T                    : {args.T}")
    print(f"Newton steps          : {args.newton_steps}")
    print(f"epochs per step       : {args.epochs_per_step}")
    print(f"batch size            : {args.batch_size}")
    print(f"network width/depth   : {args.width}/{args.depth}")
    print(f"LBFGS iters           : {args.lbfgs_iters}")
    print("=" * 72)

    # --------------------------------------------------------
    # FDM baseline
    # --------------------------------------------------------
    print("\n[1] Running FDM baseline...")
    x_fdm, t_fdm, u_fdm = fdm_semimplicit(
        nx=args.fdm_nx,
        nt=args.fdm_nt,
        final_time=args.T,
        a=args.a,
        device=device,
        dtype=torch.float64,
    )
    fdm_metrics = discrete_error_metrics(u_num=u_fdm, x=x_fdm, t_grid=t_fdm, a=args.a)

    # --------------------------------------------------------
    # Newton--PINN training
    # --------------------------------------------------------
    print("\n[2] Training Newton--PINN...")
    model = NewtonPINN(a=args.a, final_time=args.T).to(device=device, dtype=pinn_dtype)

    x_eval = torch.arange(args.eval_nx, device=device, dtype=torch.float64) * (DOMAIN_LENGTH / args.eval_nx)
    t_eval = torch.linspace(0.0, args.T, args.eval_nt + 1, device=device, dtype=torch.float64)

    metrics_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    # K=0 row
    u0_grid = eval_pinn_on_grid(
        model=model,
        x=x_eval,
        t_grid=t_eval,
        device=device,
        dtype=pinn_dtype,
        upto=0,
    ).to(dtype=torch.float64)
    m0 = discrete_error_metrics(u_num=u0_grid, x=x_eval, t_grid=t_eval, a=args.a)
    r0 = estimate_final_nonlinear_residual(
        model=model,
        upto=0,
        n_mc=max(5000, args.mc_eval // 4),
        device=device,
        dtype=pinn_dtype,
    )
    metrics_rows.append({
        "method": "Newton-PINN",
        "K": 0,
        "L2_QT": m0["L2_QT"],
        "Energy": m0["Energy"],
        "Final_L2x": m0["Final_L2x"],
        "Residual_L2": r0,
        "rho_l2": float("nan"),
        "quad_l2": float("nan"),
        "Certificate": float("nan"),
        "Energy_over_Cert": float("nan"),
    })

    step_records: list[StepRecord] = []
    for k in range(args.newton_steps):
        record = train_newton_step(
            model=model,
            step=k,
            epochs=args.epochs_per_step,
            batch_size=args.batch_size,
            lr=args.lr,
            width=args.width,
            depth=args.depth,
            lbfgs_iters=args.lbfgs_iters,
            lbfgs_batch_size=args.lbfgs_batch_size,
            device=device,
            dtype=pinn_dtype,
            print_every=args.print_every,
        )
        step_records.append(record)

        # Record a compact loss history point sequence by reusing final losses.
        # This keeps the CSV small. Full per-epoch histories are intentionally
        # not stored to avoid large files.
        training_rows.append({
            "step": k,
            "epoch": args.epochs_per_step,
            "loss": record.adam_final_loss,
            "phase": "Adam",
            "seconds": record.train_seconds,
        })
        if record.lbfgs_final_loss is not None:
            training_rows.append({
                "step": k,
                "epoch": args.epochs_per_step + args.lbfgs_iters,
                "loss": record.lbfgs_final_loss,
                "phase": "LBFGS",
                "seconds": record.train_seconds,
            })

        print(f"[step {k:02d}] Evaluating error and certificate...")
        uK_grid = eval_pinn_on_grid(
            model=model,
            x=x_eval,
            t_grid=t_eval,
            device=device,
            dtype=pinn_dtype,
            upto=k + 1,
        ).to(dtype=torch.float64)

        mK = discrete_error_metrics(u_num=uK_grid, x=x_eval, t_grid=t_eval, a=args.a)
        cert = estimate_certificate_for_correction(
            model=model,
            correction_index=k,
            n_mc=args.mc_eval,
            device=device,
            dtype=pinn_dtype,
        )
        residual_K = estimate_final_nonlinear_residual(
            model=model,
            upto=k + 1,
            n_mc=args.mc_eval,
            device=device,
            dtype=pinn_dtype,
        )

        metrics_rows.append({
            "method": "Newton-PINN",
            "K": k + 1,
            "L2_QT": mK["L2_QT"],
            "Energy": mK["Energy"],
            "Final_L2x": mK["Final_L2x"],
            "Residual_L2": residual_K,
            "rho_l2": cert["rho_l2"],
            "quad_l2": cert["hx_sq_l2"],
            "Certificate": cert["certificate"],
            "Energy_over_Cert": mK["Energy"] / cert["certificate"] if cert["certificate"] > 0 else float("nan"),
        })

    # Add FDM baseline row.
    metrics_rows.insert(0, {
        "method": "FDM",
        "K": "",
        "L2_QT": fdm_metrics["L2_QT"],
        "Energy": fdm_metrics["Energy"],
        "Final_L2x": fdm_metrics["Final_L2x"],
        "Residual_L2": float("nan"),
        "rho_l2": float("nan"),
        "quad_l2": float("nan"),
        "Certificate": float("nan"),
        "Energy_over_Cert": float("nan"),
    })

    print_metrics_table(metrics_rows)

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------
    print("[3] Saving tables...")
    write_csv(table_dir / "metrics.csv", metrics_rows)
    write_csv(table_dir / "training_summary.csv", training_rows)

    with (outdir / "run_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Experiment 1: FDM vs Newton--PINN\n")
        f.write(f"time: {now_string()}\n")
        f.write(f"device: {device}\n")
        f.write(f"torch_version: {torch.__version__}\n")
        f.write(f"preset: {args.preset}\n")
        f.write(f"a: {args.a}\n")
        f.write(f"T: {args.T}\n")
        f.write(f"newton_steps: {args.newton_steps}\n")
        f.write(f"epochs_per_step: {args.epochs_per_step}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"width: {args.width}\n")
        f.write(f"depth: {args.depth}\n")
        f.write(f"lbfgs_iters: {args.lbfgs_iters}\n")
        f.write("\nMetrics are saved in tables/metrics.csv.\n")
        f.write("Figures are saved in figures/ as PDF and PNG.\n")

    # --------------------------------------------------------
    # Final grid evaluation for figures
    # --------------------------------------------------------
    print("[4] Saving figures...")
    final_u_pinn = eval_pinn_on_grid(
        model=model,
        x=x_eval,
        t_grid=t_eval,
        device=device,
        dtype=pinn_dtype,
        upto=args.newton_steps,
    ).to(dtype=torch.float64)

    plot_final_profile(
        fig_dir=fig_dir,
        x_eval=x_eval,
        t_eval=t_eval,
        u_pinn=final_u_pinn,
        x_fdm=x_fdm,
        t_fdm=t_fdm,
        u_fdm=u_fdm,
        a=args.a,
    )
    plot_error_vs_step(fig_dir=fig_dir, metrics_rows=metrics_rows)
    plot_certificate_ratio(fig_dir=fig_dir, metrics_rows=metrics_rows)
    plot_training_loss(fig_dir=fig_dir, loss_history=training_rows)
    plot_spacetime_error(fig_dir=fig_dir, x_eval=x_eval, t_eval=t_eval, u_pinn=final_u_pinn, a=args.a)
    plot_final_error_profile(fig_dir=fig_dir, x_eval=x_eval, t_eval=t_eval, u_pinn=final_u_pinn, a=args.a)

    print("\nDone.")
    print(f"Outputs saved to: {outdir.resolve()}")
    print("Main files:")
    print(f"  {table_dir / 'metrics.csv'}")
    print(f"  {fig_dir / 'fig01_final_time_profile.pdf'}")
    print(f"  {fig_dir / 'fig02_error_and_certificate.pdf'}")
    print(f"  {fig_dir / 'fig03_certificate_ratio.pdf'}")
    print(f"  {fig_dir / 'fig05_spacetime_error.pdf'}")


if __name__ == "__main__":
    main()
