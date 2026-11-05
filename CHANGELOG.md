# Changelog

## [1.1.0]: fixes after the dissertation

These changes alter the simulated plant. The stored runs 10, 31 and 37 were produced by `v1.0.0`; their folders in `results/` are unchanged snapshots. MATLAB was not available while these changes were made: each one was checked by re-simulating in Python (`tools/selftest.py`) and by static analysis, but not run in MATLAB.

### Fixed
- **Swapped Pendubot parameters** (`test_drive/PendubotModelConf.m`). `map_2_assign_value` gave `l_1` the value of m1, `m_1` the value of l1, `l_2` the value of m2 and `m_2` the value of l2. Every parameter now receives its own value.
- **Friction** (`src/+model/Pendubot.m`). `[mu_1;mu_2]' * q_dot` subtracted one scalar from both joint equations; viscous friction is now `diag([mu_1, mu_2]) * q_dot`.
- **Unsaturated teacher torques** (`src/perturbedTrajectoriesTbD.m`). The perturbed demonstration trajectories now use the same ±`torque_sat` limit as the nominal one. Before, they set a torque range of about ±70–100 N m for the network output.
- **Torque normalisation saved per run** (`scripts/main.m`). Each new `results/<n>/` folder now gets a copy of `normalization_data.mat`. For runs 10, 31 and 37 the ranges were recovered from their rollouts and added (README, section 4.5).
- **`fepOptimizer.m`.** It used `print(...)` where `warning(...)` was meant.
- **`visualization.m`.** The cost and flatness history plots were never drawn after training: `isfield` was called on an optimiser object, where `isprop` is needed.
- **`testurdf2casadiwrapperOnMatlab.m`.** It called `model.urdf2casadiWrapperConf()`; it now calls `urdf2casadiWrapperModelConf()`.
- **Clear errors for unavailable features.** `data_generation_selector = 'basin'` needs `perturbedBasinTbD.m`, which is not part of the project, and integral flatness (`type_of_flat.integral = true`) is not supported by the optimisers. Both now stop with an explanation.

### Consequences
- On the corrected plant, none of the stored controllers is stable, including the warm starts (README, section 4.2). The committed `normalization_data.mat` and `optimal_param_teaching.mat` were made on the old plant, so run teaching by demonstration again before training.

### Added
- `LICENSE` (GPL-3.0), `CITATION.cff`, `CHANGELOG.md`, `.gitattributes`, `docs/CODE_OVERVIEW.md`.
- `tools/selftest.py` and the GitHub Actions workflow `.github/workflows/verify.yml`.
- `--plant as-run | corrected` in `tools/pendubot_sim.py` and `tools/reproduce_results.py`.

### Removed
- `docs/code_overview.txt`. Its paths were out of date and some descriptions were wrong; it is replaced by `docs/CODE_OVERVIEW.md`.

## [1.0.0]: code as used for the dissertation

- MATLAB/CasADi code, the three reported runs (10 = FEP, 31 = Adam, 37 = Yama), original figures and Python analysis tools.
- MATLAB code organised into `scripts/` (entry points) and `src/` (library). The scripts locate the repository themselves instead of relying on `pwd`; nothing else was changed.
