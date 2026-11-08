#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unforced periodic viscous Hamilton--Jacobi benchmark with a Cole--Hopf solution.

The initial heat variable is bounded below by 0.2. Its time-dependent signal
has RMS scale O(d**(-1/2)); this benchmark measures residual monitoring as
the signal decreases with dimension."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from dim_sweep_newton_pinn import (
    get_device,
    MLP,
    NewtonPINN,
    DirectPINN,
    ResourceMonitor,
    estimate_certificate_rms,
    estimate_residual_rms,
    evaluate_error_metrics,
    initial_condition_check,
    metrics_row,
    scaled_params,
    set_seed,
    train_direct_pinn,
    train_newton_step,
)

TWO_PI = 2.0 * math.pi


class UnforcedCole:
    """SweepProblem-compatible unforced quadratic viscous HJ problem."""

    def __init__(self, d: int, a: float = 0.9, b: float = 0.9,
                 final_time: float = 1.0):
        if d < 5:
            raise ValueError("d >= 5 required")
        if abs(a) + abs(b) >= 2.0:
            raise ValueError("|a|+|b| < 2 required for positivity of phi")
        self.d = d
        self.a = a
        self.b = b
        self.family = "unforced"
        self.name = f"unforced_{d}d"
        self.final_time = final_time

    # --- domain / features (same ansatz as the sweep) ---
    def sample_x(self, *, n, device, dtype):
        return TWO_PI * torch.rand(n, self.d, device=device, dtype=dtype)

    def features(self, x, t):
        tau = t / self.final_time
        return torch.cat([torch.sin(x), torch.cos(x), tau], dim=1)

    @property
    def feature_dim(self):
        return 2 * self.d + 1

    # --- Hamiltonian H(p) = |p|^2 / 2 ---
    def H(self, p):
        return 0.5 * torch.sum(p * p, dim=1, keepdim=True)

    def Hp(self, p):
        return p

    @property
    def m2_bound(self):
        return 1.0  # Hessian of |p|^2/2 is the identity

    # --- exact solution via Cole--Hopf ---
    def _phi_parts(self, x, t):
        x1 = torch.roll(x, shifts=-1, dims=1)
        x2 = torch.roll(x, shifts=-2, dims=1)
        w1 = x + x1              # modes |k|^2 = 2
        w2 = x + 2.0 * x1 - x2   # modes |k|^2 = 6
        e2 = torch.exp(-2.0 * t)
        e6 = torch.exp(-6.0 * t)
        c1 = self.a / self.d
        c2 = self.b / self.d
        phi = (2.0
               + c1 * e2 * torch.sum(torch.cos(w1), dim=1, keepdim=True)
               + c2 * e6 * torch.sum(torch.cos(w2), dim=1, keepdim=True))
        # phi_t
        phi_t = (-2.0 * c1 * e2 * torch.sum(torch.cos(w1), dim=1, keepdim=True)
                 - 6.0 * c2 * e6 * torch.sum(torch.cos(w2), dim=1, keepdim=True))
        # D phi: x_i appears in w1_i (coef 1), w1_{i-1} (coef 1);
        #        w2_i (coef 1), w2_{i-1} (coef 2), w2_{i-2} (coef -1)
        s1 = torch.sin(w1)
        s2 = torch.sin(w2)
        dphi = (-c1 * e2 * (s1 + torch.roll(s1, shifts=1, dims=1))
                - c2 * e6 * (s2 + 2.0 * torch.roll(s2, shifts=1, dims=1)
                             - torch.roll(s2, shifts=2, dims=1)))
        # Delta phi = phi_t (phi solves the heat equation): use closed form
        lap_phi = phi_t
        return phi, phi_t, dphi, lap_phi

    def exact_components(self, x, t):
        phi, phi_t, dphi, lap_phi = self._phi_parts(x, t)
        u = -2.0 * torch.log(phi)
        grad = -2.0 * dphi / phi
        # Delta u = -2 lap_phi / phi + 2 |Dphi|^2 / phi^2
        dphi2 = torch.sum(dphi * dphi, dim=1, keepdim=True)
        lap = -2.0 * lap_phi / phi + 2.0 * dphi2 / (phi * phi)
        forcing = torch.zeros_like(u)
        return u, grad, lap, forcing

    def initial_condition(self, x):
        t0 = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return self.exact_components(x, t0)[0]

    def forcing(self, x, t):
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)


def verify_unforced(problem: UnforcedCole, device, n: int = 256) -> None:
    """float64 check: closed-form Du/Lap against autograd, and R[u_ex] = 0."""
    from dim_sweep_newton_pinn import autograd_grad, laplacian_from_grad

    dtype = torch.float64
    x = problem.sample_x(n=n, device=device, dtype=dtype)
    t = problem.final_time * torch.rand(n, 1, device=device, dtype=dtype)
    x.requires_grad_(True)
    t.requires_grad_(True)
    u, grad_cf, lap_cf, _ = problem.exact_components(x, t)
    grad_auto = autograd_grad(u, x)
    lap_auto = laplacian_from_grad(grad_auto, x)
    ut_auto = autograd_grad(u, t)
    err_grad = float(torch.max(torch.abs(grad_auto - grad_cf)))
    err_lap = float(torch.max(torch.abs(lap_auto - lap_cf)))
    residual = ut_auto + problem.H(grad_auto) - lap_auto
    err_res = float(torch.max(torch.abs(residual)))
    err = max(err_grad, err_lap, err_res)
    print(f"[selfcheck] {problem.name}: |Du err|={err_grad:.2e} "
          f"|Lap err|={err_lap:.2e} |R[u_ex]|={err_res:.2e}")
    if err > 1e-8:
        raise RuntimeError(f"unforced self-check failed: {err:.3e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dims", type=int, nargs="+", default=[10, 50, 100])
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    p.add_argument("--a", type=float, default=0.9)
    p.add_argument("--b", type=float, default=0.9)
    p.add_argument("--newton_steps", type=int, default=4)
    p.add_argument("--epochs_per_step", type=int, default=4000)
    p.add_argument("--with_direct", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--eval_time_slices", type=int, default=31)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--outdir", type=str, default="reruns/phase3_unforced")
    p.add_argument("--save_models", action="store_true")
    # scaled_params compatibility
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--mc_eval", type=int, default=None)
    p.add_argument("--eval_space_samples", type=int, default=None)
    args = p.parse_args()

    device = get_device(args.device)
    dtype = torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | dims {args.dims} seeds {args.seeds}")

    rows = []
    t0_all = time.time()
    for d in args.dims:
        problem = UnforcedCole(d, a=args.a, b=args.b)
        verify_unforced(problem, device)
        sp = scaled_params(d, args)
        for seed in args.seeds:
            # ----- Newton-PINN -----
            set_seed(seed)
            model = NewtonPINN(problem).to(device=device, dtype=dtype)
            ic0 = initial_condition_check(model=model, problem=problem,
                                          device=device, dtype=dtype)
            err0 = evaluate_error_metrics(
                model=model, problem=problem, upto=0, n_space=sp["n_space"],
                n_time=args.eval_time_slices, batch_size=sp["eval_batch"],
                device=device, dtype=dtype)
            res0 = estimate_residual_rms(
                model=model, problem=problem, upto=0, n_mc=sp["mc_eval"],
                batch_size=sp["eval_batch"], device=device, dtype=dtype)
            rows.append(metrics_row(
                problem=problem, seed=seed, method="Newton-PINN", K=0,
                err=err0, residual_rms=res0, cert=None, ic_err=ic0,
                train_seconds=float("nan"), cum_train_seconds=0.0,
                peak_cuda_mb=float("nan")))
            cum = 0.0
            for k in range(args.newton_steps):
                net, _, res = train_newton_step(
                    model=model, step=k, epochs=args.epochs_per_step,
                    batch_size=sp["train_batch"], lr=args.lr,
                    width=args.width, depth=args.depth, device=device,
                    dtype=dtype, log_every=args.log_every)
                model.corrections.append(net)
                cum += res["seconds"]
                errk = evaluate_error_metrics(
                    model=model, problem=problem, upto=None,
                    n_space=sp["n_space"], n_time=args.eval_time_slices,
                    batch_size=sp["eval_batch"], device=device, dtype=dtype)
                resk = estimate_residual_rms(
                    model=model, problem=problem, upto=None,
                    n_mc=sp["mc_eval"], batch_size=sp["eval_batch"],
                    device=device, dtype=dtype)
                cert = estimate_certificate_rms(
                    model=model, correction_index=k, n_mc=sp["mc_eval"],
                    batch_size=sp["eval_batch"], device=device, dtype=dtype)
                row = metrics_row(
                    problem=problem, seed=seed, method="Newton-PINN", K=k + 1,
                    err=errk, residual_rms=resk, cert=cert, ic_err=ic0,
                    train_seconds=res["seconds"], cum_train_seconds=cum,
                    peak_cuda_mb=res["cuda_peak_allocated_mb"])
                rows.append(row)
                print(f"[{problem.name} seed={seed} K={k+1}] "
                      f"E={row['Energy_RMS']:.3e} C={row['Certificate_RMS']:.3e} "
                      f"ratio={row['Energy_over_Cert']:.3f}")
            if args.save_models:
                torch.save({"cfg": {"d": d, "a": args.a, "b": args.b},
                            "state_dict": model.state_dict()},
                           outdir / "models" / f"newton_unforced_{d}d_seed{seed}.pt")

            # ----- Direct baseline at equal total epochs -----
            set_seed(seed)
            dmodel = DirectPINN(problem, width=args.width, depth=args.depth
                                ).to(device=device, dtype=dtype)
            icd = initial_condition_check(model=dmodel, problem=problem,
                                          device=device, dtype=dtype)
            _, dres = train_direct_pinn(
                model=dmodel, epochs=args.newton_steps * args.epochs_per_step,
                batch_size=sp["train_batch"], lr=args.lr, device=device,
                dtype=dtype, log_every=args.log_every)
            errd = evaluate_error_metrics(
                model=dmodel, problem=problem, upto=None, n_space=sp["n_space"],
                n_time=args.eval_time_slices, batch_size=sp["eval_batch"],
                device=device, dtype=dtype)
            resd = estimate_residual_rms(
                model=dmodel, problem=problem, upto=None, n_mc=sp["mc_eval"],
                batch_size=sp["eval_batch"], device=device, dtype=dtype)
            rows.append(metrics_row(
                problem=problem, seed=seed, method="Direct-PINN",
                K=args.newton_steps * args.epochs_per_step, err=errd,
                residual_rms=resd, cert=None, ic_err=icd,
                train_seconds=dres["seconds"], cum_train_seconds=dres["seconds"],
                peak_cuda_mb=dres["cuda_peak_allocated_mb"]))
            print(f"[{problem.name} seed={seed} direct] E={errd['Energy_RMS']:.3e}")

            with open(outdir / "tables" / "metrics.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print(f"Done in {(time.time()-t0_all)/60:.1f} min. Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
