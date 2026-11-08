# Newton--PINN numerical experiments

Code and recorded results for Newton iteration applied to viscous Hamilton--Jacobi equations. The experiments compare frozen Newton linearization with direct and staged nonlinear PINNs, and evaluate a residual-based error indicator.

## Installation

Use Python 3.11 or newer. Create and activate an environment, then install the dependencies:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then run:

```bash
python -m pip install -r requirements.txt
```

The recorded training runs used PyTorch 2.11.0 with CUDA 12.8 and one NVIDIA GeForce RTX 5070 with 12 GB memory. Install a PyTorch build compatible with the local GPU driver for training. CPU execution is supported for the short checks and result processing.

## Reproduce figures and tables from the saved results

Run these commands from the repository root:

```bash
python summarize_results.py --verify-archive
python make_paper_figures.py --fig_dir reruns/figures --tab_dir reruns/tables
python make_phase123_tables.py --tab-dir reruns/tables
```

The first command verifies the archived data checksums and writes `tables/results_summary.json`. The other commands rebuild seven figures and the three manuscript tables. The supplementary final-time slice figure loads the saved ring models. The baseline table is also written as `baselines_table_edited.tex` to match the manuscript input filename.

The checked-in `figures/` directory contains the manuscript figures; `tables/` contains the generated tables and numerical summary.

## Run the experiments

```bash
python run_experiments.py --list
python run_experiments.py smoke --device cpu
python run_experiments.py exp1 --dry-run
python run_experiments.py exp1
python run_experiments.py all
```

`all` runs the full configurations, including the evaluation uncertainty study after training. Full training requires substantially more time than the smoke checks. Each stage writes its command configuration, execution log, and results under `reruns/`. Existing outputs require `--overwrite`; the bundled `runs/` archive is protected by the runner.

After a complete rerun:

```bash
python make_paper_figures.py --runs_root reruns --fig_dir reruns/figures --tab_dir reruns/tables
python make_phase123_tables.py --runs-root reruns --tab-dir reruns/tables
python summarize_results.py --runs-root reruns --output reruns/tables/results_summary.json
```

## Repository layout

| Path | Contents |
| --- | --- |
| `configs/experiments.json` | Full training configurations and short checks |
| `dim_sweep_newton_pinn.py` | Chain, ring, and wide Newton/direct experiments |
| `exp1_fdm_vs_newton_pinn_paper.py` | One-dimensional Cole--Hopf problem and FDM reference |
| `exp1_schedule_ablation.py` | Equal-budget training schedules |
| `phase0_mc_convergence.py` | Evaluation sample-size and time-resolution study |
| `phase1_baselines.py` | Joint composite and sequential nonlinear baselines |
| `phase2_adaptive.py` | Indicator-driven inner and outer stopping |
| `phase3_unforced.py` | Unforced high-dimensional Cole--Hopf benchmark |
| `runs/` | Archived CSV results, training histories, and model checkpoints |
| `figures/`, `tables/` | Manuscript outputs and recomputed statistics |
| `logs/` | Original execution records with machine-specific paths removed |
| `data_manifest.csv` | File sizes and SHA-256 checksums of the archived results |

See [experiment protocols](docs/EXPERIMENTS.md), [result definitions and aggregation](docs/RESULTS.md), and [release provenance](docs/PROVENANCE.md).
