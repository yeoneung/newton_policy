#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-budget training schedules for the one-dimensional Cole--Hopf problem."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from exp1_fdm_vs_newton_pinn_paper import (
    DOMAIN_LENGTH,
    NewtonPINN,
    discrete_error_metrics,
    estimate_certificate_for_correction,
    estimate_final_nonlinear_residual,
    eval_pinn_on_grid,
    get_device,
    now_string,
    set_seed,
    train_newton_step,
    write_csv,
)

SCHEDULES: dict[str, tuple[int, int, int, int]] = {
    "uniform": (2000, 2000, 2000, 2000),
    "late": (500, 500, 1000, 6000),
    "early": (6000, 1000, 500, 500),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--outdir", type=str, default="reruns/exp1_ablation")
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--eval_nx", type=int, default=512)
    parser.add_argument("--eval_nt", type=int, default=240)
    parser.add_argument("--mc_eval", type=int, default=40000)
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true", help="Tiny run for pipeline testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    dtype = torch.float32

    schedules = SCHEDULES
    if args.smoke:
        schedules = {name: tuple(max(e // 100, 5) for e in sched) for name, sched in SCHEDULES.items()}  # type: ignore[misc]
        args.seeds = args.seeds[:1]
        args.mc_eval = 2000
        args.eval_nx = 128
        args.eval_nt = 40

    outdir = Path(args.outdir)
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Experiment: training-schedule ablation (1D Cole--Hopf)")
    print(f"time      : {now_string()}")
    print(f"device    : {device}")
    print(f"schedules : {schedules}")
    print(f"seeds     : {args.seeds}")
    print("=" * 72)

    x_eval = torch.arange(args.eval_nx, device=device, dtype=torch.float64) * (
        DOMAIN_LENGTH / args.eval_nx
    )
    t_eval = torch.linspace(0.0, args.T, args.eval_nt + 1, device=device, dtype=torch.float64)

    rows: list[dict[str, Any]] = []
    t_run0 = time.time()

    for schedule_name, epochs_list in schedules.items():
        for seed in args.seeds:
            set_seed(seed)
            model = NewtonPINN(a=args.a, final_time=args.T).to(device=device, dtype=dtype)
            print(f"\n--- schedule={schedule_name} {epochs_list}, seed={seed} ---")

            rho_sum = 0.0
            cert_sum = 0.0
            for k, epochs in enumerate(epochs_list):
                record = train_newton_step(
                    model=model,
                    step=k,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    width=args.width,
                    depth=args.depth,
                    lbfgs_iters=0,
                    lbfgs_batch_size=args.batch_size,
                    device=device,
                    dtype=dtype,
                    print_every=args.print_every,
                )

                uK = eval_pinn_on_grid(
                    model=model,
                    x=x_eval,
                    t_grid=t_eval,
                    device=device,
                    dtype=dtype,
                    upto=k + 1,
                ).to(dtype=torch.float64)
                mK = discrete_error_metrics(u_num=uK, x=x_eval, t_grid=t_eval, a=args.a)
                cert = estimate_certificate_for_correction(
                    model=model,
                    correction_index=k,
                    n_mc=args.mc_eval,
                    device=device,
                    dtype=dtype,
                )
                resid = estimate_final_nonlinear_residual(
                    model=model,
                    upto=k + 1,
                    n_mc=args.mc_eval,
                    device=device,
                    dtype=dtype,
                )
                rho_sum += cert["rho_l2"]
                cert_sum += cert["certificate"]

                rows.append(
                    {
                        "schedule": schedule_name,
                        "seed": seed,
                        "K": k + 1,
                        "epochs_this_step": epochs,
                        "Energy": mK["Energy"],
                        "L2_QT": mK["L2_QT"],
                        "Final_L2x": mK["Final_L2x"],
                        "Residual_L2": resid,
                        "rho_l2": cert["rho_l2"],
                        "quad_l2": cert["hx_sq_l2"],
                        "Certificate": cert["certificate"],
                        "Energy_over_Cert": mK["Energy"] / cert["certificate"]
                        if cert["certificate"] > 0
                        else float("nan"),
                        "cum_rho_l2": rho_sum,
                        "cum_certificate": cert_sum,
                        "train_seconds": record.train_seconds,
                    }
                )
                print(
                    f"[{schedule_name}, seed={seed}, K={k + 1}] "
                    f"E={mK['Energy']:.3e}, C={cert['certificate']:.3e}, "
                    f"rho={cert['rho_l2']:.3e}, cum_rho={rho_sum:.3e}"
                )

            write_csv(table_dir / "ablation_metrics.csv", rows)

    with (outdir / "run_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Training-schedule ablation (1D Cole--Hopf Newton--PINN)\n")
        f.write(f"time: {now_string()}\n")
        f.write(f"schedules: {schedules}\n")
        f.write(f"seeds: {args.seeds}\n")
        f.write(f"batch_size: {args.batch_size}, width/depth: {args.width}/{args.depth}\n")
        f.write(f"total wall clock: {time.time() - t_run0:.1f} s\n")

    print(f"\nDone in {(time.time() - t_run0) / 60.0:.1f} min.")
    print(f"Table: {table_dir / 'ablation_metrics.csv'}")


if __name__ == "__main__":
    main()
