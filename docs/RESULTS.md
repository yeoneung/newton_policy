# Result definitions and aggregation

`summarize_results.py` recomputes the statistics in `tables/results_summary.json` directly from the archived CSV files. Standard deviations are sample standard deviations over seeds. Single-seed groups have no reported standard deviation.

## CSV fields

| Field | Meaning |
| --- | --- |
| `d` | Spatial dimension; denoted n in the manuscript |
| `K` | Number of accumulated Newton/sequential corrections; for direct methods this column stores the training epoch count |
| `Energy_RMS` | Maximum-in-time spatial mean squared error plus the time average of the spatial gradient mean squared error, followed by a square root |
| `L2_QT_RMS` | Space-time RMS solution error |
| `Residual_RMS` | Nonlinear PDE residual RMS |
| `rho_RMS` | Linearized residual RMS for the last correction |
| `quad_RMS` | Square root of the space-time mean of the fourth power of the correction gradient magnitude |
| `Certificate_RMS` | rho_RMS + (M2/2) quad_RMS |
| `Energy_over_Cert` | Energy_RMS / Certificate_RMS |
| `cum_train_seconds` | Accumulated training time through the indicated correction |
| `cuda_peak_allocated_mb` | Peak allocated CUDA memory in MiB |

Experiments 1 and 3 use physical, volume-weighted L2 and energy norms; their fields omit the `_RMS` suffix. Missing quantities are stored as `nan`.

## Cohorts

The main sweep contains 24 Newton training runs. The wide family adds 9, giving 33 runs for the K = 4 nonlinear-residual/indicator comparison. The budget and capacity-matched studies use separate two-seed cohorts. They are not pooled into the main sweep.

The main table reports K = 3 results. The direct baseline has used 16000 updates, compared with 12000 for three Newton corrections. The figure's calibration curves average the individual ratios over seeds. The comparison of the direct and Newton residuals uses ratios of seed-averaged residuals. Individual-seed ranges are identified explicitly in the summary.

## Selected recomputed statistics

| Quantity | Value |
| --- | --- |
| Ring K = 3 dimension slope, n = 10 to 50 | 1.11559 |
| Ring K = 3 dimension slope, n = 50 to 100 | 1.94509 |
| Nonlinear residual / indicator at K = 4, all 33 runs | 0.975224 to 1.005411 |
| n = 100, doubled budget, K = 4 energy error | 0.277492 |
| Late-schedule final linearized residual / uniform | 2.02906 |
| Late-schedule final linearized residual / early | 1.90575 |
| Adaptive / fixed K = 4 mean error at n = 20, tau <= 0.1, same three seeds | 1.45721 |
| Adaptive / fixed K = 4 mean error at n = 50, tau = 0.3, same three seeds | 1.77249 |

The doubled-budget space-time errors 0.037250 at n = 50 and 0.099201 at n = 100 refer to K = 3 and K = 8, respectively. Their direct comparisons use 16000 and 64000 updates.

The indicator is used empirically. Neither the stability constant nor a sampling confidence bound is evaluated. A ratio slightly above one between the sampled nonlinear residual and indicator is compatible with their separate evaluation samples.
