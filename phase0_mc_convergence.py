#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sampling and time-resolution study using the saved Newton models.

The evaluation seeds are 1001 through 1010. Residual and indicator estimates
use separate samples; spatial averages use fresh samples at each time slice."""

from __future__ import annotations

import csv
import argparse
import math
import time
from pathlib import Path

import torch

import dim_sweep_newton_pinn as sweep_mod
from dim_sweep_newton_pinn import (
    MLP,
    NewtonPINN,
    evaluate_error_metrics,
    estimate_certificate_rms,
    estimate_residual_rms,
)

# Saved checkpoints pickle classes against __main__ (the run script), so alias.
import __main__

for _name in ("ProblemConfig", "SweepProblem", "NewtonPINN", "MLP"):
    setattr(__main__, _name, getattr(sweep_mod, _name))

EVAL_SEEDS = list(range(1001, 1011))  # J = 10, fixed evaluation seeds
N_MC_LIST = [20_000, 80_000, 320_000]
TIME_SLICES = [31, 61, 121]

# (label, run_dir, family, d, seed, regime)
CASES = [
    ("resolved", "runs/sweep2_low", "ring", 10, 11, "well-resolved"),
    ("floor", "runs/sweep2_ring_high", "ring", 50, 11, "optimization-floor"),
    ("partial", "runs/sweep2_ring_high", "ring", 100, 11, "partial-resolution"),
    ("failed", "runs/sweep_wide", "wide", 20, 11, "failed"),
]

WIDTH, DEPTH = 128, 5


def eval_batch_for(d: int) -> int:
    return 2048 if d <= 20 else (1024 if d <= 50 else 512)


def load_model(run_dir: str, family: str, d: int, seed: int, device: torch.device):
    path = Path(run_dir) / "models" / f"newton_{family}_{d}d_seed{seed}.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["cfg"]
    # v1 checkpoints (wide family) carry a wrong `family` in cfg; the
    # filename is authoritative.  Rebuild the config from it.
    if cfg.family != family or cfg.d != d:
        print(f"[warn] {path.name}: checkpoint cfg says "
              f"{cfg.family}_{cfg.d}d ({payload.get('version')}); "
              f"using filename family/d instead")
        cfg = sweep_mod.ProblemConfig(
            family=family, d=d,
            final_time=getattr(cfg, "final_time", 1.0),
            beta_cross=getattr(cfg, "beta_cross", 0.35),
            gamma_cross=getattr(cfg, "gamma_cross", 0.25),
        )
    problem = sweep_mod.SweepProblem(cfg)
    model = NewtonPINN(problem).to(device)
    n_corr = len({k.split(".")[1] for k in payload["state_dict"] if k.startswith("corrections.")})
    for _ in range(n_corr):
        model.corrections.append(MLP(problem.feature_dim, WIDTH, DEPTH).to(device))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, problem, n_corr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", choices=[case[0] for case in CASES])
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    only = args.cases
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    dtype = torch.float32
    print(f"device: {device}")
    outdir = args.outdir or Path("reruns/phase0_mc_convergence")
    outdir.mkdir(parents=True, exist_ok=True)

    mc_rows, ts_rows = [], []
    # keep rows of cases not being re-run
    if only:
        import csv as _csv
        for fname, dest in (("mc_convergence.csv", mc_rows), ("timeslice.csv", ts_rows)):
            f = outdir / fname
            if f.exists():
                for r in _csv.DictReader(open(f)):
                    if r["case"] not in only:
                        dest.append({k: (float(v) if k not in
                                    ("case", "regime", "family") else v)
                                    for k, v in r.items()})

    for label, run_dir, family, d, seed, regime in CASES:
        if only and label not in only:
            continue
        model, problem, n_corr = load_model(
            args.runs_root / Path(run_dir).relative_to("runs"), family, d, seed, device)
        bsz = eval_batch_for(d)
        print(f"=== {label}: {family} d={d} seed={seed} ({regime}), K={n_corr} ===")

        # --- (i)+(ii): eval-seed replicates x MC sample sizes -------------
        for n_mc in N_MC_LIST:
            n_space = max(512, n_mc // 31)
            for es in EVAL_SEEDS:
                torch.manual_seed(es)
                t0 = time.perf_counter()
                err = evaluate_error_metrics(
                    model=model, problem=problem, upto=None,
                    n_space=n_space, n_time=31, batch_size=bsz,
                    device=device, dtype=dtype,
                )
                res = estimate_residual_rms(
                    model=model, problem=problem, upto=None,
                    n_mc=n_mc, batch_size=bsz, device=device, dtype=dtype,
                )
                cert = estimate_certificate_rms(
                    model=model, correction_index=n_corr - 1,
                    n_mc=n_mc, batch_size=bsz, device=device, dtype=dtype,
                )
                mc_rows.append({
                    "case": label, "regime": regime, "family": family, "d": d,
                    "train_seed": seed, "eval_seed": es, "n_mc": n_mc,
                    "n_space": n_space,
                    "Energy_RMS": err["Energy_RMS"],
                    "L2_QT_RMS": err["L2_QT_RMS"],
                    "Residual_RMS": res,
                    "Certificate_RMS": cert["Certificate_RMS"],
                    "rho_RMS": cert["rho_RMS"],
                    "quad_RMS": cert["quad_RMS"],
                    "eval_seconds": time.perf_counter() - t0,
                })
            es_vals = [r for r in mc_rows if r["case"] == label and r["n_mc"] == n_mc]
            for col in ("Energy_RMS", "Certificate_RMS", "Residual_RMS"):
                xs = [r[col] for r in es_vals]
                m = sum(xs) / len(xs)
                s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
                print(f"  n_mc={n_mc:>7d} {col:>16s}: mean={m:.4e} rel_std={100*s/m:.2f}%")

        # --- (iii): time-slice refinement (protocol n_space, 3 seeds) -----
        n_space_proto = 8192 if d <= 20 else (4096 if d <= 50 else 2048)
        for n_time in TIME_SLICES:
            for es in EVAL_SEEDS[:3]:
                torch.manual_seed(es)
                err = evaluate_error_metrics(
                    model=model, problem=problem, upto=None,
                    n_space=n_space_proto, n_time=n_time, batch_size=bsz,
                    device=device, dtype=dtype,
                )
                ts_rows.append({
                    "case": label, "family": family, "d": d, "train_seed": seed,
                    "eval_seed": es, "n_time": n_time, "n_space": n_space_proto,
                    "Energy_RMS": err["Energy_RMS"],
                    "L2_QT_RMS": err["L2_QT_RMS"],
                })
            xs = [r["Energy_RMS"] for r in ts_rows
                  if r["case"] == label and r["n_time"] == n_time]
            print(f"  n_time={n_time:>4d} Energy_RMS mean={sum(xs)/len(xs):.4e}")

    with open(outdir / "mc_convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mc_rows[0].keys()))
        w.writeheader()
        w.writerows(mc_rows)
    with open(outdir / "timeslice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()))
        w.writeheader()
        w.writerows(ts_rows)
    print(f"saved -> {outdir}")


if __name__ == "__main__":
    main()
