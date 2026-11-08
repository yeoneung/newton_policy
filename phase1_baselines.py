#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capacity-matched direct and sequential nonlinear PINN baselines."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from dim_sweep_newton_pinn import (
    get_device,
    MLP,
    NewtonPINN,
    ProblemConfig,
    ResourceMonitor,
    SweepProblem,
    estimate_residual_rms,
    evaluate_error_metrics,
    freeze_module,
    initial_condition_check,
    metrics_row,
    nonlinear_residual_and_grad,
    probe_residual_rms,
    sample_xt,
    scaled_params,
    set_seed,
    verify_exact,
)


class CompositeDirectPINN(torch.nn.Module):
    """C2: g + t * sum_j N_j, all subnetworks trained jointly (no staging)."""

    def __init__(self, problem: SweepProblem, width: int, depth: int, n_nets: int):
        super().__init__()
        self.problem = problem
        self.nets = torch.nn.ModuleList(
            MLP(problem.feature_dim, width, depth) for _ in range(n_nets)
        )

    def u(self, x, t, upto=None):
        del upto
        out = self.problem.initial_condition(x)
        feats = self.problem.features(x, t)
        for net in self.nets:
            out = out + t * net(feats)
        return out


def train_nonlinear(
    *, model, epochs, batch_size, lr, device, dtype, log_every, params, tag
):
    """Adam on the (preconditioned) nonlinear residual of `model`."""
    optimizer = torch.optim.Adam(params, lr=lr)
    sigma = max(
        probe_residual_rms(
            model=model, upto=None, n=4 * batch_size, batch_size=batch_size,
            device=device, dtype=dtype,
        ),
        1e-8,
    )
    print(f"[{model.problem.name}, {tag}] residual scale sigma={sigma:.4e}")
    with ResourceMonitor(device) as mon:
        for ep in range(1, epochs + 1):
            x, t = sample_xt(n=batch_size, problem=model.problem, device=device, dtype=dtype)
            residual, _ = nonlinear_residual_and_grad(model, x, t, upto=None)
            loss = torch.mean((residual / sigma) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ep == 1 or ep % log_every == 0 or ep == epochs:
                snap = mon.snapshot()
                print(
                    f"[{model.problem.name}, {tag}] epoch {ep:05d}/{epochs}, "
                    f"loss(norm)={float(loss.detach().cpu()):.4e}, "
                    f"time={snap['seconds']:.1f}s, "
                    f"peak_cuda={snap['cuda_peak_allocated_mb']:.0f}MB"
                )
        resources = mon.snapshot()
    return resources


def eval_row(*, model, problem, seed, method, K, sp, args, device, dtype,
             ic_err, train_seconds, cum_seconds, peak_mb):
    err = evaluate_error_metrics(
        model=model, problem=problem, upto=None,
        n_space=sp["n_space"], n_time=args.eval_time_slices,
        batch_size=sp["eval_batch"], device=device, dtype=dtype,
    )
    res = estimate_residual_rms(
        model=model, problem=problem, upto=None,
        n_mc=sp["mc_eval"], batch_size=sp["eval_batch"], device=device, dtype=dtype,
    )
    return metrics_row(
        problem=problem, seed=seed, method=method, K=K, err=err,
        residual_rms=res, cert=None, ic_err=ic_err,
        train_seconds=train_seconds, cum_train_seconds=cum_seconds,
        peak_cuda_mb=peak_mb,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--families", type=str, nargs="+", default=["ring"])
    p.add_argument("--dims", type=int, nargs="+", default=[50])
    p.add_argument("--seeds", type=int, nargs="+", default=[11])
    p.add_argument("--methods", type=str, nargs="+", default=["seq", "comp"],
                   choices=["seq", "comp"])
    p.add_argument("--newton_steps", type=int, default=4)
    p.add_argument("--epochs_per_step", type=int, default=4000)
    p.add_argument("--comp_epochs", type=int, default=16000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--beta_cross", type=float, default=0.35)
    p.add_argument("--gamma_cross", type=float, default=0.25)
    p.add_argument("--eval_time_slices", type=int, default=31)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--outdir", type=str, default="reruns/phase1_baselines")
    p.add_argument("--save_models", action="store_true")
    # scaled_params reads these attributes; keep auto behaviour
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
    print(f"device: {device} | methods: {args.methods} | outdir: {outdir.resolve()}")

    all_rows = []
    t0_all = time.time()
    for family in args.families:
        for d in args.dims:
            cfg = ProblemConfig(
                family=family, d=d, final_time=args.T,
                beta_cross=args.beta_cross, gamma_cross=args.gamma_cross,
            )
            problem = SweepProblem(cfg)
            verify_exact(problem, device)
            sp = scaled_params(d, args)
            for seed in args.seeds:
                # ---------------- C3: sequential nonlinear correction ------
                if "seq" in args.methods:
                    set_seed(seed)
                    model = NewtonPINN(problem).to(device=device, dtype=dtype)
                    ic0 = initial_condition_check(
                        model=model, problem=problem, device=device, dtype=dtype)
                    cum = 0.0
                    for k in range(args.newton_steps):
                        freeze_module(model)
                        net = MLP(problem.feature_dim, args.width, args.depth).to(
                            device=device, dtype=dtype)
                        model.corrections.append(net)
                        res = train_nonlinear(
                            model=model, epochs=args.epochs_per_step,
                            batch_size=sp["train_batch"], lr=args.lr,
                            device=device, dtype=dtype, log_every=args.log_every,
                            params=net.parameters(),
                            tag=f"seq k={k} seed={seed}",
                        )
                        freeze_module(net)
                        cum += res["seconds"]
                        row = eval_row(
                            model=model, problem=problem, seed=seed,
                            method="Seq-Nonlinear", K=k + 1, sp=sp, args=args,
                            device=device, dtype=dtype, ic_err=ic0,
                            train_seconds=res["seconds"], cum_seconds=cum,
                            peak_mb=res["cuda_peak_allocated_mb"],
                        )
                        all_rows.append(row)
                        print(f"[seq {problem.name} seed={seed} K={k+1}] "
                              f"E={row['Energy_RMS']:.3e} R={row['Residual_RMS']:.3e}")
                    if args.save_models:
                        torch.save(
                            {"cfg": cfg, "state_dict": model.state_dict()},
                            outdir / "models" / f"seq_{family}_{d}d_seed{seed}.pt",
                        )

                # ---------------- C2: matched-capacity joint direct --------
                if "comp" in args.methods:
                    set_seed(seed)
                    model = CompositeDirectPINN(
                        problem, args.width, args.depth, args.newton_steps
                    ).to(device=device, dtype=dtype)
                    ic0 = initial_condition_check(
                        model=model, problem=problem, device=device, dtype=dtype)
                    res = train_nonlinear(
                        model=model, epochs=args.comp_epochs,
                        batch_size=sp["train_batch"], lr=args.lr,
                        device=device, dtype=dtype, log_every=args.log_every,
                        params=model.parameters(), tag=f"comp seed={seed}",
                    )
                    row = eval_row(
                        model=model, problem=problem, seed=seed,
                        method="Composite-Direct", K=args.comp_epochs, sp=sp,
                        args=args, device=device, dtype=dtype, ic_err=ic0,
                        train_seconds=res["seconds"], cum_seconds=res["seconds"],
                        peak_mb=res["cuda_peak_allocated_mb"],
                    )
                    all_rows.append(row)
                    print(f"[comp {problem.name} seed={seed}] "
                          f"E={row['Energy_RMS']:.3e} R={row['Residual_RMS']:.3e}")
                    if args.save_models:
                        torch.save(
                            {"cfg": cfg, "state_dict": model.state_dict()},
                            outdir / "models" / f"comp_{family}_{d}d_seed{seed}.pt",
                        )

                # incremental write so partial runs are usable
                with open(outdir / "tables" / "metrics.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                    w.writeheader()
                    w.writerows(all_rows)

    print(f"Done in {(time.time()-t0_all)/60:.1f} min. Outputs: {outdir.resolve()}")


if __name__ == "__main__":
    main()
