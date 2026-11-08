# Experiment protocols

The complete commands are specified in `configs/experiments.json`. `run_experiments.py` resolves the output directory and executes those commands without changing the training objectives.

All problems are periodic with final time T = 1. Each correction has the form t N(sin x, cos x, t), so the initial condition and spatial periodicity are imposed exactly. A fresh uniform collocation batch is drawn at every Adam update.

## Main configurations

| Experiment | Stages | Configuration |
| --- | --- | --- |
| One-dimensional Cole--Hopf | `exp1` | Five corrections; 4000 Adam updates and 400 strong-Wolfe L-BFGS iterations per correction; width 96, depth 5; float64 |
| Main sweep, n = 10, 20 | `main-low` | Chain and ring; seeds 11, 12, 13 |
| Main sweep, n = 50, 100 | `main-chain-high`, `main-ring-high`, `main-ring50-extra`, `main-ring100-extra` | Chain seed 11; ring seeds 11 through 15 |
| Growing interaction order | `wide` | n = 5, 10, 20; seeds 11, 12, 13 |
| More corrections | `budget-lowmid`, `budget100` | Eight corrections, 4000 Adam updates each; seeds 11, 12 |
| More training per correction | `budget50-double`, `budget100-double` | 8000 Adam updates per correction; K = 4 at n = 50 and K = 8 at n = 100 |
| Capacity and staging | `baseline50`, `baseline100` | Ring, n = 50, 100; seeds 11, 12; four networks |
| Schedule ablation | `ablation` | Seeds 7, 8, 9; four corrections and 8000 total Adam updates |
| Adaptive iteration | `adaptive-low`, `adaptive50` | Ring; three seeds; tolerances 0.3, 0.1, 0.05 at n = 10, 20 and 0.3 at n = 50 |
| Unforced problem | `unforced` | n = 10, 50, 100; seeds 11, 12; four corrections |
| Sampling uncertainty | `mc` | Four saved models; evaluation seeds 1001 through 1010 |

The standard high-dimensional configuration uses width 128, depth 5, float32, and four corrections of 4000 Adam updates. The single-network direct baseline uses 16000 updates. Training batches contain 4096 points for n <= 20, 2048 for n = 50, and 1024 for n = 100. The learning rate is 0.001. CUDA runs of the dimension sweep enable TF32 matrix multiplication.

High-dimensional Newton losses are divided by a positive residual scale fixed during each inner solve. Experiments 1 and 3 use the unscaled squared linearized residual. Reported residuals and indicators use their original units.

## Evaluation

| Dimension | Residual/indicator space-time samples | Spatial samples per time slice |
| --- | --- | --- |
| n <= 20 | 80000 | 8192 |
| n = 50 | 40000 | 4096 |
| n = 100 | 20000 | 2048 |

The energy error uses 31 uniform time slices and fresh spatial samples at each slice. Time integration uses the trapezoidal rule. Error, nonlinear residual, and indicator evaluations draw separate samples. Validation estimates used inside the adaptive controller have their own sample counts; see `phase2_adaptive.py`.

The one-dimensional release configuration uses 512 spatial evaluation points and 241 time levels, with periodic central differences for the error gradient. The FDM reference uses 512 spatial grid points and 4000 time steps. The grid parameters are explicit in the released command; the original one-dimensional run summary did not record the evaluation grid.

The sampling study varies space-time sample counts over 20000, 80000, and 320000. Its energy evaluation uses max(512, floor(N/31)) spatial points at each of 31 time slices. A separate refinement study uses 31, 61, and 121 time slices with the protocol spatial sample count.

## Adaptive controller

The controller evaluates the forcing condition after each block of 500 updates, with eta = 0.5 and at most 8000 updates per correction. It tests the correction at scales 1, 0.5, and 0.25, accepting the first scale that reduces the sampled nonlinear residual. The validation sample is fixed by seed and independent of the training batches. The forcing test precedes damping.

The indicator is recomputed for the accepted correction. The iteration stops at the requested indicator threshold, a relative indicator decrease smaller than or equal to 2 percent, a rejected correction, or six accepted corrections. True errors are used only for reporting.

## Capacity-matched comparisons

C2 jointly optimizes four networks on the nonlinear residual. C3 adds four networks sequentially, holding previous networks fixed and training each stage on the nonlinear residual of the entire approximation. Both use 16000 total Adam updates. Their comparisons with Newton use the same two seeds, rather than the five-seed main-sweep averages at n = 50 and 100.

The extra-budget direct baseline at n = 100 uses 64000 updates. At n = 50 the recorded single-network direct comparison uses 16000 updates. Epoch matching, wall-time matching, and parameter matching are distinct comparisons.
