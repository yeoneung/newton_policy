#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the manuscript figures and main sweep table from saved results."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# IO helpers
# =============================================================================


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out: dict[str, Any] = {}
            for k, v in row.items():
                if v is None or v == "":
                    out[k] = float("nan")
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": True,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_dir / (stem + '.pdf')}")


def _paren_digits(mean: float, std: float) -> Optional[tuple[int, int]]:
    """Pick decimals so the standard deviation fits in 1-2 parenthesized digits
    of the last quoted place (NIST-style "1.85(2)"). None if it cannot."""
    for dec in (2, 3, 1, 0):
        digits = int(round(std * 10**dec))
        if 1 <= digits <= 99:
            return dec, digits
    return None


def fmt_sci_latex(mean: float, std: Optional[float] = None) -> str:
    if not np.isfinite(mean):
        return "--"
    if mean == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(mean))))
    mant = mean / 10**exp
    if std is not None and np.isfinite(std) and std > 0:
        picked = _paren_digits(mant, std / 10**exp)
        if picked is not None:
            dec, digits = picked
            return f"${mant:.{dec}f}({digits}){{\\times}}10^{{{exp}}}$"
    return f"${mant:.2f}{{\\times}}10^{{{exp}}}$"


def fmt_ratio_latex(mean: float, std: Optional[float] = None) -> str:
    if not np.isfinite(mean):
        return "--"
    if std is not None and np.isfinite(std) and std > 0:
        picked = _paren_digits(mean, std)
        if picked is not None:
            dec, digits = picked
            return f"${mean:.{dec}f}({digits})$"
    return f"${mean:.2f}$"


# =============================================================================
# Data aggregation
# =============================================================================


def collect_metrics(run_dirs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rd in run_dirs:
        path = rd / "tables" / "metrics.csv"
        found = load_csv(path)
        if found:
            rows.extend(found)
            print(f"[data] {path}: {len(found)} rows")
        else:
            print(f"[warn] missing or empty: {path}")
    return rows


def frows(rows: list[dict[str, Any]], family: str, method: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("family") == family and r.get("method") == method]


def mean_std(
    rows: list[dict[str, Any]], key: str, *, d: int, K: Optional[int] = None
) -> tuple[float, float, int]:
    vals = [
        float(r[key])
        for r in rows
        if int(r["d"]) == d and (K is None or int(r["K"]) == K) and np.isfinite(float(r[key]))
    ]
    if not vals:
        return float("nan"), float("nan"), 0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, len(arr)


def dims_present(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({int(r["d"]) for r in rows})


# =============================================================================
# Figure: main sweep panel
# =============================================================================


def _sweep_common(sweep):
    ring_n = frows(sweep, "ring", "Newton-PINN")
    ring_d = frows(sweep, "ring", "Direct-PINN")
    chain_n = frows(sweep, "chain", "Newton-PINN")
    chain_d = frows(sweep, "chain", "Direct-PINN")
    main_n = ring_n if ring_n else chain_n
    dims = dims_present(main_n)
    all_dims = sorted(set(dims_present(ring_n) + dims_present(chain_n)))
    colors = {d: plt.cm.viridis(v) for d, v in zip(all_dims, np.linspace(0.05, 0.85, len(all_dims)))}
    Ks = sorted({int(r["K"]) for r in main_n})
    return ring_n, ring_d, chain_n, chain_d, main_n, dims, all_dims, colors, Ks


def fig_sweep_main(
    sweep: list[dict[str, Any]], fig_dir: Path, *, K_main: int,
) -> None:
    """Main sweep figure: (a) error and indicator vs d, (b) calibration."""
    ring_n, ring_d, chain_n, chain_d, main_n, dims, all_dims, colors, Ks = _sweep_common(sweep)
    if not ring_n and not chain_n:
        print("[warn] no sweep data; skipping fig_sweep_main")
        return

    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    # (a) error and indicator vs d at K_main; direct baseline for reference
    ax = axes[0]
    for fam_n, fam_d, mk, col, name in (
        (ring_n, ring_d, "o", "C0", "ring"),
        (chain_n, chain_d, "s", "C3", "chain"),
    ):
        if not fam_n:
            continue
        dd = dims_present(fam_n)
        e = np.array([mean_std(fam_n, "Energy_RMS", d=d, K=K_main) for d in dd])
        c = np.array([mean_std(fam_n, "Certificate_RMS", d=d, K=K_main) for d in dd])
        ax.errorbar(dd, e[:, 0], yerr=e[:, 1], fmt=mk + "-", color=col, capsize=3,
                    label=rf"{name}, $E_{{{K_main}}}$")
        ax.errorbar(dd, c[:, 0], yerr=c[:, 1], fmt=mk + "--", color=col, alpha=0.65,
                    markersize=4, capsize=3,
                    label=rf"{name}, $\mathfrak{{C}}_{{{K_main}}}$")
        if fam_d:
            ddd = dims_present(fam_d)
            ed = np.array([mean_std(fam_d, "Energy_RMS", d=d) for d in ddd])
            ax.errorbar(ddd, ed[:, 0], yerr=ed[:, 1], fmt=mk + ":", color=col, alpha=0.5,
                        markerfacecolor="none", capsize=3,
                        label=f"{name}, direct $E$")
    e0 = mean_std(main_n, "Energy_RMS", d=dims[0], K=0)[0]
    if np.isfinite(e0):
        ax.axhline(e0, color="0.3", linewidth=0.8, linestyle=":")
        ax.text(all_dims[0], e0 * 1.08, "initial guess $E_0$", fontsize=7, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_dims)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel("RMS-normalized error and indicator")
    ax.set_title(rf"(a) $E_{{{K_main}}}$ (solid), $\mathfrak{{C}}_{{{K_main}}}$ (dashed), direct (dotted)")
    ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=2)

    # (b) calibration ratio vs K, both families
    ax = axes[1]
    for d in dims_present(ring_n):
        r_mean = [mean_std(ring_n, "Energy_over_Cert", d=d, K=k)[0] for k in Ks if k >= 1]
        ax.plot([k for k in Ks if k >= 1], r_mean, "o-", color=colors[d], label=f"ring $d={d}$")
    for d in dims_present(chain_n):
        r_mean = [mean_std(chain_n, "Energy_over_Cert", d=d, K=k)[0] for k in Ks if k >= 1]
        ax.plot(
            [k for k in Ks if k >= 1], r_mean, "^--", color=colors[d], alpha=0.8,
            label=f"chain $d={d}$",
        )
    ax.axhline(1.0, color="0.3", linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 1.1)
    ax.set_xticks([k for k in Ks if k >= 1])
    ax.set_xlabel("Newton--PINN step $K$")
    ax.set_ylabel(r"$E_K/\mathfrak{C}_K$")
    ax.set_title("(b) empirical indicator calibration")
    ax.legend(frameon=False, ncol=2, fontsize=7)

    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_sweep_main")


def fig_sweep_budget(
    sweep: list[dict[str, Any]], fig_dir: Path, *,
    ext: Optional[list[dict[str, Any]]] = None,
    ext2: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Saturation and budget dependence: E_K, C_K vs K for the ring family,
    continued to K=8 and with the doubled per-step budget."""
    ring_n, ring_d, chain_n, chain_d, main_n, dims, all_dims, colors, Ks = _sweep_common(sweep)
    if not main_n:
        print("[warn] no sweep data; skipping fig_sweep_budget")
        return

    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4))
    for d in dims:
        e_mean = [mean_std(main_n, "Energy_RMS", d=d, K=k)[0] for k in Ks]
        c_mean = [mean_std(main_n, "Certificate_RMS", d=d, K=k)[0] for k in Ks]
        ax.semilogy(Ks, e_mean, "o-", color=colors[d], label=f"$d={d}$")
        kc = [k for k, c in zip(Ks, c_mean) if np.isfinite(c)]
        cc = [c for c in c_mean if np.isfinite(c)]
        ax.semilogy(kc, cc, "s--", color=colors[d], alpha=0.65, markersize=4)
    ext_n = frows(ext or [], "ring", "Newton-PINN")
    for d in dims_present(ext_n):
        if d not in colors:
            continue
        eKs = sorted({int(r["K"]) for r in ext_n if int(r["d"]) == d and int(r["K"]) > max(Ks)})
        if not eKs:
            continue
        seg = [max(Ks)] + eKs
        e_mean = [mean_std(ext_n, "Energy_RMS", d=d, K=k)[0] for k in seg]
        c_mean = [mean_std(ext_n, "Certificate_RMS", d=d, K=k)[0] for k in seg]
        ax.semilogy(seg, e_mean, "o-", color=colors[d])
        ax.semilogy(seg, c_mean, "s--", color=colors[d], alpha=0.65, markersize=4)
    ext2_n = frows(ext2 or [], "ring", "Newton-PINN")
    for d in dims_present(ext2_n):
        if d not in colors:
            continue
        e2Ks = sorted({int(r["K"]) for r in ext2_n if int(r["d"]) == d})
        e_mean = [mean_std(ext2_n, "Energy_RMS", d=d, K=k)[0] for k in e2Ks]
        ax.semilogy(e2Ks, e_mean, "D-.", color=colors[d], markersize=4,
                    label=rf"$d={d}$, $2\times$ ep/step")
    ax.set_xlabel("Newton--PINN step $K$")
    ax.set_ylabel("RMS-normalized error and indicator")
    ax.set_title(r"ring family: $E_K$ (solid), $\mathfrak{C}_K$ (dashed)")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_sweep_budget")


def fig_sweep_cost(sweep: list[dict[str, Any]], fig_dir: Path, *, K_main: int) -> None:
    """Computational cost diagnostic: training time and peak memory vs d."""
    ring_n, ring_d, chain_n, chain_d, main_n, dims, all_dims, colors, Ks = _sweep_common(sweep)
    if not main_n:
        print("[warn] no sweep data; skipping fig_sweep_cost")
        return
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.4))
    src_n, src_d = (ring_n, ring_d) if ring_n else (chain_n, chain_d)
    dd = dims_present(src_n)
    t_n = [mean_std(src_n, "cum_train_seconds", d=d, K=K_main)[0] for d in dd]
    ax.plot(dd, t_n, "o-", label=f"Newton train time ($K={K_main}$)")
    if src_d:
        ddd = dims_present(src_d)
        t_d = [mean_std(src_d, "train_seconds", d=d)[0] for d in ddd]
        ax.plot(ddd, t_d, "s--", label="Direct train time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(all_dims)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("dimension $d$")
    ax.set_ylabel("training seconds")
    ax.set_title("computational cost (ring)")
    ax2 = ax.twinx()
    m_n = [
        max((float(r["cuda_peak_allocated_mb"]) for r in src_n
             if int(r["d"]) == d and np.isfinite(float(r["cuda_peak_allocated_mb"]))),
            default=float("nan"))
        for d in dd
    ]
    ax2.plot(dd, m_n, "d-.", color="0.4", label="peak CUDA MB")
    ax2.set_ylabel("peak CUDA memory (MB)", color="0.35")
    ax2.tick_params(axis="y", colors="0.35")
    ax2.grid(False)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_sweep_cost")


# =============================================================================
# Figure: wide family (growing interaction order)
# =============================================================================


def fig_wide(wide: list[dict[str, Any]], fig_dir: Path, *, K_main: int) -> None:
    wide_n = frows(wide, "wide", "Newton-PINN")
    wide_d = frows(wide, "wide", "Direct-PINN")
    if not wide_n:
        print("[warn] no wide-family data; skipping fig_wide")
        return
    dims = dims_present(wide_n)
    orders = [(d + 1) // 2 for d in dims]  # phase interaction order

    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))

    # (a) errors vs interaction order
    ax = axes[0]
    e0 = [mean_std(wide_n, "Energy_RMS", d=d, K=0)[0] for d in dims]
    eK = np.array([mean_std(wide_n, "Energy_RMS", d=d, K=K_main) for d in dims])
    ax.semilogy(orders, e0, "o--", color="0.45", label="$E_0$ (initial guess)")
    ax.errorbar(orders, eK[:, 0], yerr=eK[:, 1], fmt="o-", color="C0", capsize=3,
                label=f"$E_{{{K_main}}}$, Newton")
    if wide_d:
        eD = np.array([mean_std(wide_d, "Energy_RMS", d=d) for d in dims])
        ax.errorbar(orders, eD[:, 0], yerr=eD[:, 1], fmt="s--", color="C3", capsize=3,
                    label="direct PINN")
    ax.set_xticks(orders)
    ax.set_xticklabels([f"{p}\n($d={d}$)" for p, d in zip(orders, dims)])
    ax.set_xlabel("phase interaction order")
    ax.set_ylabel("energy error (RMS)")
    ax.set_title("(a) representational breakdown")
    ax.legend(frameon=False, fontsize=8)

    # (b) ratio vs K stays calibrated even under failure
    ax = axes[1]
    Ks = sorted({int(r["K"]) for r in wide_n if int(r["K"]) >= 1})
    colors = {d: plt.cm.viridis(v) for d, v in zip(dims, np.linspace(0.1, 0.8, len(dims)))}
    for d in dims:
        r_mean = [mean_std(wide_n, "Energy_over_Cert", d=d, K=k)[0] for k in Ks]
        ax.plot(Ks, r_mean, "o-", color=colors[d], label=f"$d={d}$")
    ax.axhline(1.0, color="0.3", linewidth=0.8, linestyle=":")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Newton--PINN step $K$")
    ax.set_ylabel(r"$E_K/\mathfrak{C}_K$")
    ax.set_title("(b) indicator remains large under failure")
    ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_wide")


# =============================================================================
# Figure: slice heatmaps (ring family, regenerated from saved models)
# =============================================================================


def fig_sweep_slices(
    run_dirs: list[Path], fig_dir: Path, *, family: str, dims: list[int], seed: int,
    width: int, depth: int, slice_n: int
) -> None:
    try:
        import torch

        import dim_sweep_newton_pinn as sweep_mod
        from dim_sweep_newton_pinn import MLP, NewtonPINN, ProblemConfig, SweepProblem, TWO_PI
    except Exception as exc:  # pragma: no cover
        print(f"[warn] cannot import sweep module for slices: {exc}")
        return

    # Checkpoints were saved by dim_sweep_newton_pinn.py run as a script, so
    # pickled classes resolve against __main__; alias them there for torch.load.
    import __main__

    for _name in ("ProblemConfig", "SweepProblem", "NewtonPINN", "MLP"):
        setattr(__main__, _name, getattr(sweep_mod, _name))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    found: list[tuple[int, Any, Any]] = []
    for d in dims:
        model_path = None
        for rd in run_dirs:
            cand = rd / "models" / f"newton_{family}_{d}d_seed{seed}.pt"
            if cand.exists():
                model_path = cand
                break
        if model_path is None:
            print(f"[warn] no saved model for {family}_{d}d seed {seed}")
            continue
        payload = torch.load(model_path, map_location=device, weights_only=False)
        cfg: ProblemConfig = payload["cfg"]
        problem = SweepProblem(cfg)
        model = NewtonPINN(problem).to(device)
        n_corr = len({k.split(".")[1] for k in payload["state_dict"].keys() if k.startswith("corrections.")})
        for _ in range(n_corr):
            model.corrections.append(MLP(problem.feature_dim, width, depth).to(device))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        found.append((d, problem, model))

    if not found:
        return

    setup_matplotlib()
    fig, axes = plt.subplots(1, len(found), figsize=(4.1 * len(found), 3.6))
    if len(found) == 1:
        axes = [axes]
    for ax, (d, problem, model) in zip(axes, found):
        grid = torch.linspace(0.0, TWO_PI, slice_n, device=device)
        X1, X2 = torch.meshgrid(grid, grid, indexing="ij")
        x = torch.full((slice_n * slice_n, problem.d), 0.37 * TWO_PI, device=device)
        x[:, 0] = X1.reshape(-1)
        x[:, 1] = X2.reshape(-1)
        t = torch.full((slice_n * slice_n, 1), problem.final_time, device=device)
        with torch.no_grad():
            pred = model.u(x, t, upto=None)
            exact = problem.exact_components(x, t)[0]
            err = torch.abs(pred - exact).reshape(slice_n, slice_n).cpu().numpy()
            # time-dependent signal on this slice: the part the corrections
            # must represent (the initial condition is exact).  Reported as an
            # RMS, and only per panel: the slice is a measure-zero section of
            # T^d at one value of the remaining coordinates, so its signal
            # amplitude is not the global O(1) scale and the ratio of the two
            # is not comparable across d.  The peak-to-RMS ratio of the error
            # is comparable, and is what the panel is used for.
            signal_rms = float((exact - problem.initial_condition(x)).pow(2).mean().sqrt())
        err_rms = float(np.sqrt((err**2).mean()))
        print(
            f"[slice] d={d:<4d} max_err={err.max():.4f} rms_err={err_rms:.4f} "
            f"peak/rms={err.max()/err_rms:.2f} slice_signal_rms={signal_rms:.4f}"
        )
        im = ax.imshow(err, origin="lower", aspect="auto", extent=[0, 2 * np.pi, 0, 2 * np.pi])
        fig.colorbar(im, ax=ax, shrink=0.9)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(
            rf"$d={d}$: $\max|\widehat u^K-u_{{\rm ex}}|={err.max():.3f}$, "
            rf"peak/RMS $={err.max()/err_rms:.1f}$"
        )
        ax.grid(False)
    fig.suptitle(
        rf"Final-time slice error $|\widehat u^K - u_{{\rm ex}}|$ ({family} family)", y=1.02
    )
    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_sweep_slices")


# =============================================================================
# Figure: exp1 panel
# =============================================================================


def fig_exp1_main(exp1_dir: Path, fig_dir: Path) -> None:
    rows = load_csv(exp1_dir / "tables" / "metrics.csv")
    if not rows:
        print("[warn] no exp1 convergence data; skipping fig_exp1_main")
        return
    nrows = [r for r in rows if r.get("method") == "Newton-PINN"]
    K = np.array([float(r["K"]) for r in nrows])
    order = np.argsort(K)
    K = K[order]
    E = np.array([float(r["Energy"]) for r in nrows])[order]
    C = np.array([float(r["Certificate"]) for r in nrows])[order]
    ratio = np.array([float(r["Energy_over_Cert"]) for r in nrows])[order]

    setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))

    ax = axes[0]
    ax.semilogy(K, E, "o-", label=r"true error $E_K$")
    fin = np.isfinite(C)
    ax.semilogy(K[fin], C[fin], "s--", label=r"residual indicator $\mathfrak{C}_K$")
    ax.set_xlabel("Newton--PINN step $K$")
    ax.set_ylabel("energy norm")
    ax.set_title("(a) error and residual indicator")
    ax.legend(frameon=False)

    ax = axes[1]
    Ek = E[:-1]
    Ek1 = E[1:]
    ax.loglog(Ek, Ek1, "o-", label=r"$(E_K, E_{K+1})$")
    span = np.array([Ek.min() * 0.5, Ek.max() * 2.0])
    ref = Ek1[0] / Ek[0] ** 2 * span**2
    ax.loglog(span, ref, "--", color="0.4", label=r"slope 2 (quadratic)")
    ax.set_xlabel(r"$E_K$")
    ax.set_ylabel(r"$E_{K+1}$")
    ax.set_title("(b) pre-floor contraction check")
    ax.legend(frameon=False)

    ax = axes[2]
    fin = np.isfinite(ratio)
    ax.plot(K[fin], ratio[fin], "o-")
    ax.set_xlabel("Newton--PINN step $K$")
    ax.set_ylabel(r"$E_K/\mathfrak{C}_K$")
    ax.set_title("(c) calibration ratio")

    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_exp1_main")


# =============================================================================
# Figure: schedule ablation
# =============================================================================


def fig_ablation(abl_dir: Path, fig_dir: Path) -> None:
    rows = load_csv(abl_dir / "tables" / "ablation_metrics.csv")
    if not rows:
        print("[warn] no ablation data; skipping fig_ablation")
        return
    K_final = max(int(r["K"]) for r in rows)
    finals = [r for r in rows if int(r["K"]) == K_final]
    schedules = [s for s in ("uniform", "late", "early") if any(r["schedule"] == s for r in finals)]

    def agg(sched: str, key: str) -> tuple[float, float]:
        vals = np.array([float(r[key]) for r in finals if r["schedule"] == sched])
        return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    xpos = np.arange(len(schedules))
    w = 0.27
    for off, key, label in (
        (-w, "Energy", r"final error $E_4$"),
        (0.0, "Certificate", r"final residual indicator $\mathfrak{C}_4$"),
        (w, "cum_rho_l2", r"$\sum_{k<4}\|\rho_k\|$"),
    ):
        means = [agg(s, key)[0] for s in schedules]
        stds = [agg(s, key)[1] for s in schedules]
        ax.bar(xpos + off, means, width=w * 0.92, yerr=stds, capsize=3, label=label)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(schedules)
    ax.set_title("(a) equal budget, different schedules")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for sched, marker in zip(schedules, ("o", "s", "^")):
        e = np.array([float(r["Energy"]) for r in finals if r["schedule"] == sched])
        c = np.array([float(r["Certificate"]) for r in finals if r["schedule"] == sched])
        s = np.array([float(r["cum_rho_l2"]) for r in finals if r["schedule"] == sched])
        ax.loglog(c, e, marker, color="C0", label=f"{sched}: vs $\\mathfrak{{C}}_4$")
        ax.loglog(s, e, marker, markerfacecolor="none", color="C3",
                  label=f"{sched}: vs $\\sum\\|\\rho_k\\|$")
    lims = np.array([min(ax.get_xlim()[0], ax.get_ylim()[0]),
                     max(ax.get_xlim()[1], ax.get_ylim()[1])])
    ax.loglog(lims, lims, ":", color="0.5", linewidth=0.8)
    ax.set_xlabel("indicator value")
    ax.set_ylabel(r"final error $E_4$")
    ax.set_title("(b) what predicts the final error")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.tight_layout()
    save_figure(fig, fig_dir, "fig_ablation")


# =============================================================================
# LaTeX table (chain + ring)
# =============================================================================


def sweep_table(sweep: list[dict[str, Any]], tab_dir: Path, *, K_main: int) -> None:
    tab_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("% Auto-generated by make_paper_figures.py -- do not edit by hand.")
    lines.append("\\begin{tabular}{lrcccccccc}")
    lines.append("\\toprule")
    lines.append(
        f"family & $d$ & $E_{{{K_main}}}$ & $\\mathfrak{{C}}_{{{K_main}}}$ "
        f"& $E_{{{K_main}}}/\\mathfrak{{C}}_{{{K_main}}}$ "
        f"& $\\|\\mathcal R[\\widehat u^{{{K_main}}}]\\|$ "
        f"& $E^{{\\rm direct}}$ & $\\|\\mathcal R[v^{{\\rm direct}}]\\|$ "
        f"& $t_{{K={K_main}}}$ (s) & mem (GB) \\\\"
    )
    any_rows = False
    for family in ("chain", "ring"):
        fam_n = frows(sweep, family, "Newton-PINN")
        fam_d = frows(sweep, family, "Direct-PINN")
        if not fam_n:
            continue
        lines.append("\\midrule")
        for d in dims_present(fam_n):
            e_m, e_s, n = mean_std(fam_n, "Energy_RMS", d=d, K=K_main)
            c_m, c_s, _ = mean_std(fam_n, "Certificate_RMS", d=d, K=K_main)
            r_m, r_s, _ = mean_std(fam_n, "Energy_over_Cert", d=d, K=K_main)
            rn_m, rn_s, _ = mean_std(fam_n, "Residual_RMS", d=d, K=K_main)
            ed_m, ed_s, nd = mean_std(fam_d, "Energy_RMS", d=d)
            rd_m, rd_s, _ = mean_std(fam_d, "Residual_RMS", d=d)
            t_m, _, _ = mean_std(fam_n, "cum_train_seconds", d=d, K=K_main)
            mem_vals = [
                float(r["cuda_peak_allocated_mb"]) for r in fam_n
                if int(r["d"]) == d and np.isfinite(float(r["cuda_peak_allocated_mb"]))
            ]
            mem = max(mem_vals) / 1024.0 if mem_vals else float("nan")
            ratio_str = fmt_ratio_latex(r_m, r_s if n > 1 else None)
            lines.append(
                f"{family} & {d} & {fmt_sci_latex(e_m, e_s if n > 1 else None)} "
                f"& {fmt_sci_latex(c_m, c_s if n > 1 else None)} "
                f"& {ratio_str} "
                f"& {fmt_sci_latex(rn_m, rn_s if n > 1 else None)} "
                f"& {fmt_sci_latex(ed_m, ed_s if nd > 1 else None)} "
                f"& {fmt_sci_latex(rd_m, rd_s if nd > 1 else None)} "
                f"& {t_m:.0f} & {mem:.2f} \\\\"
            )
            any_rows = True
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if not any_rows:
        print("[warn] no sweep data; table not written")
        return
    out = tab_dir / "sweep_table.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_root", type=str, default="runs")
    parser.add_argument("--fig_dir", type=str, default="figures")
    parser.add_argument("--tab_dir", type=str, default="tables")
    parser.add_argument("--K_main", type=int, default=3)
    parser.add_argument("--slice_family", type=str, default="ring")
    parser.add_argument("--slice_dims", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--slice_seed", type=int, default=11)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--slice_n", type=int, default=160)
    args = parser.parse_args()

    runs = Path(args.runs_root)
    fig_dir = Path(args.fig_dir)
    tab_dir = Path(args.tab_dir)
    sweep_dirs = [runs / "sweep2_low", runs / "sweep2_ring_high", runs / "sweep2_chain_high",
                  # additional ring seeds at d=50,100
                  runs / "sweep2_ring50_s1315", runs / "sweep2_ring100_s1315"]
    wide_dirs = [runs / "sweep_wide"]
    # supplementary K=8 extensions (panel (a) only)
    ext_dirs = [runs / "sweep2_ring100_K8", runs / "sweep2_ring_lowmid_K8"]
    # extended-budget runs (8000 ep/step); d=50 is carried only to K=4, see
    # sec_numerics.tex -- its per-step gain has already saturated there.
    ext2_dirs = [runs / "sweep2_ring100_K8_e8k", runs / "sweep2_ring50_e8k"]

    sweep = collect_metrics(sweep_dirs)
    ext = collect_metrics([d for d in ext_dirs if (d / "tables" / "metrics.csv").exists()])
    ext2 = collect_metrics([d for d in ext2_dirs if (d / "tables" / "metrics.csv").exists()])
    if sweep:
        fig_sweep_main(sweep, fig_dir, K_main=args.K_main)
        fig_sweep_budget(sweep, fig_dir, ext=ext, ext2=ext2)
        fig_sweep_cost(sweep, fig_dir, K_main=args.K_main)
        sweep_table(sweep, tab_dir, K_main=args.K_main)
        fig_sweep_slices(
            sweep_dirs, fig_dir, family=args.slice_family, dims=args.slice_dims,
            seed=args.slice_seed, width=args.width, depth=args.depth, slice_n=args.slice_n,
        )
    wide = collect_metrics(wide_dirs)
    fig_wide(wide, fig_dir, K_main=args.K_main)
    fig_exp1_main(runs / "exp1_convergence", fig_dir)
    fig_ablation(runs / "exp1_ablation", fig_dir)
    print("Done.")


if __name__ == "__main__":
    main()
