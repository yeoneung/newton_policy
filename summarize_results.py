#!/usr/bin/env python3
"""Recompute manuscript statistics from the archived result tables."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

import numpy as np

from make_phase123_tables import adaptive_rows

ROOT = Path(__file__).resolve().parent
MAIN_RUNS = (
    "sweep2_low", "sweep2_chain_high", "sweep2_ring_high",
    "sweep2_ring50_s1315", "sweep2_ring100_s1315",
)


def read(root, name, filename="metrics.csv"):
    with (root / name / "tables" / filename).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def describe(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    if not values:
        return None
    return {"mean": st.mean(values), "std": st.stdev(values) if len(values) > 1 else None,
            "min": min(values), "max": max(values), "count": len(values)}


def grouped(rows, fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in fields)].append(row)
    result = []
    for key, selected in sorted(groups.items()):
        entry = dict(zip(fields, key))
        entry["statistics"] = {field: describe(r[field] for r in selected)
                               for field in metrics}
        result.append(entry)
    return result


def verify_archive():
    with (ROOT / "data_manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        path = ROOT / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"] or path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Archive checksum mismatch: {row['path']}")
    print(f"Verified {len(rows)} archived result files")


def summarize(root):
    main = [row for name in MAIN_RUNS for row in read(root, name)]
    keys = [(r["family"], r["d"], r["seed"], r["method"], r["K"]) for r in main]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate main-sweep rows")
    newton = [r for r in main if r["method"] == "Newton-PINN"]
    later = [r for r in newton if 2 <= int(r["K"]) <= 4]
    last = [r for r in newton + read(root, "sweep_wide")
            if r["method"] == "Newton-PINN" and int(r["K"]) == 4]
    slopes = {}
    for family in ("chain", "ring"):
        E = {d: st.mean(float(r["Energy_RMS"]) for r in newton
                        if r["family"] == family and int(r["d"]) == d and int(r["K"]) == 3)
             for d in (10, 50, 100)}
        slopes[family] = {"10_to_50": math.log(E[50] / E[10]) / math.log(5),
                          "50_to_100": math.log(E[100] / E[50]) / math.log(2)}

    budgets = []
    for name, epochs in (("sweep2_ring100_K8", 4000), ("sweep2_ring100_K8_e8k", 8000)):
        rows = read(root, name)
        for k in range(1, 9):
            E = st.mean(float(r["Energy_RMS"]) for r in rows
                        if r["method"] == "Newton-PINN" and int(r["K"]) == k)
            budgets.append({"epochs_per_step": epochs, "K": k, "total_epochs": epochs * k, "Energy_RMS": E})
    logB = np.log([r["total_epochs"] for r in budgets])
    logE = np.log([r["Energy_RMS"] for r in budgets])
    exponent, intercept = np.polyfit(logB, logE, 1)
    prediction = exponent * logB + intercept
    R2 = 1 - float(np.sum((logE - prediction) ** 2) / np.sum((logE - logE.mean()) ** 2))

    adaptive = adaptive_rows(root)
    adaptive_summary = []
    by_case = defaultdict(list)
    for row in adaptive:
        by_case[(int(row["d"]), float(row["tau"]))].append(row)
    for (d, tau), rows in sorted(by_case.items()):
        fixed = []
        for row in rows:
            reference = next(r for r in newton if r["family"] == "ring" and int(r["d"]) == d
                             and r["seed"] == row["seed"] and int(r["K"]) == 4)
            fixed.append(float(reference["Energy_RMS"]))
        adaptive_summary.append({
            "n": d, "tau": tau, "seeds": [int(r["seed"]) for r in rows],
            "Energy_RMS": describe(r["Energy_RMS"] for r in rows),
            "Certificate_RMS": describe(r["Certificate_RMS"] for r in rows),
            "epochs_used": describe(r["epochs_used"] for r in rows),
            "mean_error_ratio_to_same_seed_K4": st.mean(float(r["Energy_RMS"]) for r in rows) / st.mean(fixed),
        })

    ablation = [r for r in read(root, "exp1_ablation", "ablation_metrics.csv") if int(r["K"]) == 4]
    rho = {schedule: st.mean(float(r["rho_l2"]) for r in ablation if r["schedule"] == schedule)
           for schedule in ("early", "uniform", "late")}
    with (root / "phase0_mc_convergence/mc_convergence.csv").open(newline="", encoding="utf-8") as stream:
        mc = list(csv.DictReader(stream))
    mc_groups = grouped(mc, ("case", "n_mc"), ("Energy_RMS", "Residual_RMS", "Certificate_RMS"))
    relative_std = {
        field: max(item["statistics"][field]["std"] / item["statistics"][field]["mean"]
                   for item in mc_groups)
        for field in ("Energy_RMS", "Residual_RMS", "Certificate_RMS")
    }
    baseline_rows = [r for d in (50, 100) for r in read(root, f"phase1_ring{d}")
                     if r["method"] == "Composite-Direct" or int(r["K"]) == 4]
    metrics = ("Energy_RMS", "L2_QT_RMS", "Residual_RMS", "Certificate_RMS",
               "Energy_over_Cert", "cum_train_seconds")
    return {
        "normalization": "Physical L2/energy norms for experiments 1 and 3; RMS normalization for high-dimensional experiments.",
        "exp1": read(root, "exp1_convergence"),
        "main_sweep": grouped(main, ("family", "d", "method", "K"), metrics),
        "main_run_count": sum(int(r["K"]) == 4 for r in newton),
        "main_dimension_slopes_K3": slopes,
        "calibration_K2_to_K4": {
            "n_le_50": describe(r["Energy_over_Cert"] for r in later if int(r["d"]) <= 50),
            "n_100": describe(r["Energy_over_Cert"] for r in later if int(r["d"]) == 100),
            "energy_max_over_min": max(float(r["Energy_RMS"]) for r in later) / min(float(r["Energy_RMS"]) for r in later),
        },
        "nonlinear_residual_over_indicator_K4": {
            "run_count": len(last),
            **describe(float(r["Residual_RMS"]) / float(r["Certificate_RMS"]) for r in last),
        },
        "n100_budget": {"points": budgets, "power_exponent": float(exponent), "log_log_R_squared": R2},
        "capacity_matched_baselines": grouped(baseline_rows, ("d", "method"), metrics),
        "ablation": {
            "final_results": grouped(ablation, ("schedule",), ("Energy", "Certificate", "rho_l2", "cum_rho_l2")),
            "late_rho_over_uniform": rho["late"] / rho["uniform"],
            "late_rho_over_early": rho["late"] / rho["early"],
        },
        "unforced": grouped(read(root, "phase3_unforced"), ("d", "method", "K"), metrics),
        "adaptive": adaptive_summary,
        "adaptive_indicator_stops": sum(r["stop_reason"] == "certified" for r in adaptive),
        "adaptive_indicator_stops_with_error_below_tau": sum(
            r["stop_reason"] == "certified" and float(r["Energy_RMS"]) <= float(r["tau"]) for r in adaptive),
        "evaluation_max_relative_std": relative_std,
    }


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("tables/results_summary.json"))
    parser.add_argument("--verify-archive", action="store_true")
    args = parser.parse_args()
    if args.verify_archive:
        verify_archive()
    result = finite_json(summarize(args.runs_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
