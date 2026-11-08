# Release provenance

The experiment sources and recorded results were taken from the manuscript experiment archive. The archived CSV files and checkpoint tensors are preserved byte for byte. `data_manifest.csv` records each archived result file's size and SHA-256 checksum. `source_manifest.json` records the input script hashes and the packaging changes.

The release adds portable output paths, a complete experiment runner, explicit configuration records for new executions, a numerical summary script, and command-line paths for table and evaluation scripts. The one-dimensional paper preset now matches the recorded five-correction, 4000-Adam, 400-L-BFGS configuration and selects float64. The default dimension list excludes the unsupported five-dimensional chain case. The PDE definitions, training residuals, optimizers, and error/indicator kernels are unchanged.

The raw results are organized by completed experiment. Partial low-dimensional restart directories were already merged into `sweep2_low`; they are not duplicated in this release. Adaptive results are consolidated in `phase2_adaptive/tables/adaptive_all.csv`; superseded tolerance trials and partial copies are excluded. Machine-specific user directory paths have been removed from the execution logs.

Saved models are included for the main sweep, wide family, budget studies, capacity-matched baselines, and unforced Newton experiments. The original archive contains no checkpoints for the one-dimensional convergence, schedule ablation, or adaptive experiments; their numerical tables and available training logs are included.

The archived PyTorch checkpoints contain the original Python configuration objects. The supplied loading code resolves their class names. Some wide-family checkpoints carry a legacy family label in that object; `phase0_mc_convergence.py` reconstructs the family and dimension from the checkpoint filename before loading the unchanged model tensors.

Original training logs record PyTorch 2.11.0 with CUDA 12.8. Release validation used Python 3.14.6, PyTorch 2.11.0+cu128, NumPy 2.4.6, Matplotlib 3.11.0, and psutil 7.2.2. Training results can vary with hardware and numerical libraries. Regenerating outputs from the archived CSV files reproduces the reported numerical inputs without repeating optimization.
