#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Indicator-driven Newton iteration with inner stopping and step damping."""

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
    ProblemConfig,
    SweepProblem,
    correction_residual,
    estimate_certificate_rms,
    estimate_residual_rms,
    evaluate_error_metrics,
    freeze_module,
    metrics_row,  # noqa: F401  (schema reference)
    nonlinear_residual_and_grad,
    probe_residual_rms,
    sample_xt,
    scaled_params,
    set_seed,
    verify_exact,
)


def validation_rho_rms(model, net, step, n, batch_size, device, dtype, seed):
    """||rho_k|| on an independent validation sample (fixed eval seed)."""
    gen_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    s2 = 0.0
    count = 0
    for start in range(0, n, batch_size):
        bsz = min(batch_size, n - start)
        x, t = sample_xt(n=bsz, problem=model.problem, device=device, dtype=dtype)
        rho = correction_residual(model=model, net=net, step=step, x=x, t=t)
        s2 += float(torch.sum(rho.detach() ** 2).cpu())
        count += bsz
    torch.random.set_rng_state(gen_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return math.sqrt(s2 / count)


class ScaledNet(torch.nn.Module):
    """lam * net(z), used for damped correction candidates."""

    def __init__(self, net: torch.nn.Module, lam: float):
        super().__init__()
        self.net = net
        self.lam = lam

    def forward(self, z):
        return self.lam * self.net(z)


def run_adaptive(*, problem, seed, tau, eta, block, max_epochs_step, max_steps,
                 sp, lr, width, depth, device, dtype, val_seed, floor_rtol,
                 log_every_blocks):
    set_seed(seed)
    model = NewtonPINN(problem).to(device=device, dtype=dtype)
    counts = {"accepted": 0, "damped": 0, "rejected": 0}
    t0 = time.perf_counter()
    epochs_used = 0
    rho_evals = 0
    cert_prev = float("inf")
    stop_reason = "max_steps"
    K_done = 0

    for k in range(max_steps):
        freeze_module(model)
        # ||R[u^k]|| for the forcing condition (validation estimate)
        R_k = estimate_residual_rms(
            model=model, problem=problem, upto=k, n_mc=sp["mc_eval"] // 2,
            batch_size=sp["eval_batch"], device=device, dtype=dtype,
        )
        net = MLP(problem.feature_dim, width, depth).to(device=device, dtype=dtype)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        sigma = max(
            probe_residual_rms(model=model, upto=k, n=4 * sp["train_batch"],
                               batch_size=sp["train_batch"], device=device,
                               dtype=dtype),
            1e-8,
        )
        # ---- inner loop: blocks until forcing condition or budget ----
        ep_step = 0
        forced = False
        n_blocks = 0
        while ep_step < max_epochs_step:
            for _ in range(block):
                x, t = sample_xt(n=sp["train_batch"], problem=problem,
                                 device=device, dtype=dtype)
                rho = correction_residual(model=model, net=net, step=k, x=x, t=t)
                loss = torch.mean((rho / sigma) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            ep_step += block
            n_blocks += 1
            rho_val = validation_rho_rms(
                model, net, k, n=sp["mc_eval"] // 4,
                batch_size=sp["eval_batch"], device=device, dtype=dtype,
                seed=val_seed,
            )
            rho_evals += 1
            if n_blocks % log_every_blocks == 0:
                print(f"    [k={k}] ep={ep_step:5d} ||rho||_val={rho_val:.4e} "
                      f"target={eta * R_k:.4e}")
            if rho_val <= eta * R_k:
                forced = True
                break
        epochs_used += ep_step

        # ---- accept / damp / reject on the nonlinear residual ----
        freeze_module(net)
        best_lam, best_R = None, R_k
        for lam in (1.0, 0.5, 0.25):
            cand = net if lam == 1.0 else ScaledNet(net, lam)
            model.corrections.append(cand)
            R_new = estimate_residual_rms(
                model=model, problem=problem, upto=k + 1,
                n_mc=sp["mc_eval"] // 2, batch_size=sp["eval_batch"],
                device=device, dtype=dtype,
            )
            model.corrections.pop(-1)
            if R_new < best_R:
                best_lam, best_R = lam, R_new
                break  # first improving lambda wins (1.0 preferred)
        if best_lam is None:
            counts["rejected"] += 1
            stop_reason = "floor_rejected"
            print(f"    [k={k}] rejected (no lambda improves R={R_k:.4e}); stop")
            break
        cand = net if best_lam == 1.0 else ScaledNet(net, best_lam)
        model.corrections.append(cand)
        counts["accepted" if best_lam == 1.0 else "damped"] += 1
        K_done = k + 1

        # ---- certificate and outer termination ----
        cert = estimate_certificate_rms(
            model=model, correction_index=k, n_mc=sp["mc_eval"],
            batch_size=sp["eval_batch"], device=device, dtype=dtype,
        )
        C = cert["Certificate_RMS"]
        print(f"  [k={k}] forced={forced} ep={ep_step} lam={best_lam} "
              f"R:{R_k:.4e}->{best_R:.4e} C={C:.4e} tau={tau}")
        if C <= tau:
            stop_reason = "certified"
            break
        if C >= (1.0 - floor_rtol) * cert_prev:
            stop_reason = "floor_certificate"
            break
        cert_prev = C

    wall = time.perf_counter() - t0
    # ---- reporting only: true error ----
    err = evaluate_error_metrics(
        model=model, problem=problem, upto=None, n_space=sp["n_space"],
        n_time=31, batch_size=sp["eval_batch"], device=device, dtype=dtype,
    )
    C_final = C if K_done > 0 else float("nan")
    return {
        "family": problem.family, "d": problem.d, "seed": seed, "tau": tau,
        "eta": eta, "K": K_done, "epochs_used": epochs_used,
        "rho_evaluations": rho_evals, "wall_seconds": wall,
        "accepted": counts["accepted"], "damped": counts["damped"],
        "rejected": counts["rejected"], "stop_reason": stop_reason,
        "Certificate_RMS": C_final, "Energy_RMS": err["Energy_RMS"],
        "L2_QT_RMS": err["L2_QT_RMS"],
        "certified_below_tau": int(stop_reason == "certified"),
        "true_error_below_tau": int(err["Energy_RMS"] <= tau),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dims", type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.1, 0.05])
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--block", type=int, default=500)
    p.add_argument("--max_epochs_step", type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=6)
    p.add_argument("--floor_rtol", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--val_seed", type=int, default=90001)
    p.add_argument("--log_every_blocks", type=int, default=4)
    p.add_argument("--outdir", type=str, default="reruns/phase2_adaptive")
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
    print(f"device: {device} | dims {args.dims} seeds {args.seeds} taus {args.taus}")

    rows = []
    for d in args.dims:
        cfg = ProblemConfig(family="ring", d=d, final_time=1.0,
                            beta_cross=0.35, gamma_cross=0.25)
        problem = SweepProblem(cfg)
        verify_exact(problem, device)
        sp = scaled_params(d, args)
        for tau in args.taus:
            for seed in args.seeds:
                print(f"=== ring d={d} seed={seed} tau={tau} ===")
                row = run_adaptive(
                    problem=problem, seed=seed, tau=tau, eta=args.eta,
                    block=args.block, max_epochs_step=args.max_epochs_step,
                    max_steps=args.max_steps, sp=sp, lr=args.lr,
                    width=args.width, depth=args.depth, device=device,
                    dtype=dtype, val_seed=args.val_seed,
                    floor_rtol=args.floor_rtol,
                    log_every_blocks=args.log_every_blocks,
                )
                rows.append(row)
                print(f"--> K={row['K']} reason={row['stop_reason']} "
                      f"E={row['Energy_RMS']:.3e} C={row['Certificate_RMS']:.3e} "
                      f"epochs={row['epochs_used']} wall={row['wall_seconds']:.0f}s")
                with open(outdir / "tables" / "adaptive.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
    print("Done.")


if __name__ == "__main__":
    main()
