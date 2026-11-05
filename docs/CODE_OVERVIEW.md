# Code overview

A file-by-file reference. For the method, results and run instructions, see the [README](../README.md).

**Used** means the file is part of the Pendubot pipeline behind the reported results. **Example** means it supports the other example models shipped with the framework. **Legacy** means it is kept for reference and is not needed to run anything.

## `scripts/` (entry points)

| File | Status | Purpose |
|---|---|---|
| `main.m` | Used | Builds the Pendubot rollout cost J and flatness B, then either trains a controller with the optimiser chosen in `experimentEnvOptConf.m` (`flag_mode = 0`) or re-tests `results/<folder_number>/optimal_param.mat` (`flag_mode = 1`). Each training run is saved to the next free `results/<n>/`, together with copies of the config, model, controller and `normalization_data.mat`. |
| `mainTeachingByDemonstration.m` | Used | Warm start. An LQR (or PID) teacher produces demonstration trajectories, and the network is fitted to the teacher torques with mini-batch Adam. Writes `test_drive/normalization_data.mat` and `test_drive/optimal_param_teaching.mat`. |

## `src/` (library; added to the MATLAB path by the scripts)

| File | Status | Purpose |
|---|---|---|
| `integra.m` | Used | One-step explicit Euler integrator as a CasADi function `F(v0, x0, r, p) -> (xf, qf)`: `x+ = x + (f(v, x, u) + r) DT`. |
| `costAndDerBuilder.m` | Used | Unrolls the closed loop for N steps symbolically. Returns `funJ(r0, v0, w0)`, the total cost (running + terminal), and `gradJ_Fun`, its gradient with respect to the network weights. Applies the output de-normalisation. |
| `flatnessBuilder.m` | Used | Flatness B of J with respect to the plant parameters v: `'2order'` = `log(1/2 (1 + ||grad_v J||^2) + (tr hess_v J)^2)`, `'1orderStd'` = a box-volume form. Returns `B_fun` and `gradB_Fun` (gradient with respect to the weights). Optional Monte Carlo integral flatness (`computeIntegralFlat`). |
| `costAndDerBuilderTeaching.m` | Used | Mean-squared error between network and teacher torques, and its gradient, for teaching by demonstration. |
| `perturbedTrajectoriesTbD.m` | Used | Teacher rollout from `xInit` (torque saturated at ±`torque_sat`) plus `number_of_traj - 1` copies whose states are perturbed with Gaussian noise. The teacher torques of the perturbed copies are not saturated. |
| `globalNormalize.m` | Used | Element-wise min-max scaling to [−1, 1]. |
| `creatingGridBoundUncertainVariable.m` | Used | Adds `low = v_nominal + lb` and `up = v_nominal + ub` to `struct_env_var`. |
| `visualization.m` | Used | Plots the nominal closed loop (`visualizationAux`), then re-simulates with `v_nominal` scaled by 1.0, 1.1, … (one run per grid point of `struct_env_var`; nine for the Pendubot). Saves `x_stack.mat`, `u_stack.mat` and `state_and_control.png`. |
| `visualizationAux.m` | Used | Simulates one closed-loop rollout and plots x1…x4 and u1. Reports instability when a state exceeds 1e4. |

### `src/+controller/`

| File | Status | Purpose |
|---|---|---|
| `baseController.m` | Used | Abstract base class: sizes, parameters, normalisation fields, collocated-dynamics helper. |
| `neuralController.m` | Used | Fully connected network built in CasADi (`layer_size`, hidden activation, output activation). Loads `normalization_data.mat` from the folder it is given and de-normalises the output. |
| `lqrController.m` | Used | LQR teacher. Linearises `x_dot` at `xDes` once, computes K with `lqr` (Control System Toolbox) and applies `tau = K (xDes - x)`. |
| `PIDLinearizedController.m` | Example | Computed-torque PD teacher (`Ki` is unused). It applies the full 2x2 mass matrix, so its output does not match the single-input Pendubot. |
| `dummyController.m` | Example | Controller whose output is its parameter vector (used with `dummyModelConf.m`). |
| `neuralController_old.m` | Legacy | Earlier network class without normalisation. |

### `src/+model/`

| File | Status | Purpose |
|---|---|---|
| `baseModel.m` | Used | Abstract base: stores x, u, v, assigns parameters through `assignValueToObjField`, builds M, C, G and `x_dot`. |
| `assignValueToObjField.m` | Used | Resolves a `containers.Map` such as `'m_2' -> 'v_1'` into object properties (`fp_i` = `fixed_parameters(i)`, `v_i` = `v(i)`). |
| `initializeXDot.m` | Used | Allocates `x_dot` with the right symbolic type. |
| `Pendubot.m` | Used | Pendubot dynamics `M q_ddot + C q_dot + G = [u; 0] - friction` (Lanari and Oriolo), with `I_i = m_i a_i^2 / 12`. |
| `cartPole.m` | Example | Cart-pole model. |
| `mobilePointMassSpringDamp.m` | Example | Two independent mass-spring-damper axes (two inputs). |
| `triRManipulator.m` | Example | Planar 3R manipulator (generated M, C, G). |
| `urdf2casadiWrapper.m` | Example | Builds CasADi dynamics from a URDF in `urdf/`. Needs urdf2casadi and its spatial_v2 helpers; the forward-dynamics routine is taken from urdf2casadi. |
| `mdlPuma.m` | Legacy | Puma 560 model from Peter Corke's Robotics Toolbox (LGPL-3.0). Needs that toolbox. |
| `generateModel.m` | Legacy | Generates `dD.m` (Puma direct dynamics) with the Symbolic Math Toolbox. |
| `testurdf2casadiwrapperOnMatlab.m` | Legacy | Compares the URDF wrapper with iDynTree. Calls a file name that does not exist (`urdf2casadiWrapperConf`). |
| `robotRobTolBox.m`, `robotRobTolBoxConf.m` | Legacy | Incomplete copies of the cart-pole model; they do not run. |
| `urdf/*.urdf` | Example | KUKA KR210 (from ROS-Industrial `kuka_experimental`), KUKA LWR (`lwr_description`) and a two-link arm. `*_env` variants replace masses and inertias with `envvar_n` placeholders, which become uncertain parameters. |

### `src/+optim/`

| File | Status | `optimization_type` | Purpose |
|---|---|---|---|
| `adamOptimizer.m` | Used | 0, 1 | Type 0: mini-batch Adam on the teaching MSE. Type 1: Adam on `grad J` only (B is logged, not used). |
| `yamaOptimizer.m` | Used | 2 | Task priority: drive J to `J_val_des` at rate `alpha_F`, and B to `B_val_des` at rate `alpha_B` in the null space of `grad J`. Fixed Euler step. Live plot of J, B, orthogonality and gradient norms. |
| `ipoptOptimizer.m` | Example | 3 | IPOPT on `0.001 (J - J_val_des)^2 + B`. |
| `yamaAdaGradOptimizer.m` | Example | 4 | Same direction as type 2, scaled by one RMS step size. |
| `fepOptimizer.m` | Used | 5 | Port of Feppon's null-space optimiser (GPL-3.0): minimise B subject to `J <= J_val_des`. Uses a qpOASES dual QP, a null-space step and a range step, with merit backtracking. |

### `src/examples/`

| File | Purpose |
|---|---|
| `cartPoleModelConf.m` | Cart-pole configuration (uncertain pole mass). |
| `massSpringDampModelConf.m` | Mass-spring-damper configuration. |
| `triRModelConf.m` | 3R manipulator configuration. |
| `urdf2casadiWrapperModelConf.m` | KUKA LWR configuration via the URDF wrapper. |
| `dummyModelConf.m`, `dummyControllerConf.m` | Analytic test function for exercising the optimisers. |

To use one, call it in `scripts/main.m` instead of `PendubotModelConf()`, and give `neuralControllerConf.m` a matching `layer_size`.

## `robotics_result/robust_optimization/test_drive/` (experiment folder)

| File | Purpose |
|---|---|
| `experimentEnvOptConf.m` | Horizon (`N = 800`, `DT = 0.01`), flatness settings and the settings of every optimiser. `optimization_type` selects the method. The committed version is the one used for run 37 (Yama). |
| `PendubotModelConf.m` | Pendubot parameters, target, cost, uncertain parameters `v = [m_2, l_2]` and sweep grid. The parameter map was corrected after the reported runs (see README, issue 1). |
| `neuralControllerConf.m` | 4-10-1 tanh network with output de-normalisation. |
| `cartPoleModelConf.m` | Copy of the example cart-pole config (whitespace differences only). |
| `normalization_data.mat` | Teacher torque range used to de-normalise the network output (written by teaching by demonstration, 23 Aug 2025). |
| `optimal_param_teaching.mat` | Warm-start weights (written by teaching by demonstration, 23 Aug 2025). |
| `results/<n>/` | One folder per training run; only 10 (FEP), 31 (Adam) and 37 (Yama) are versioned. Each holds `init_param.mat`, `optimal_param.mat`, `param_stack.mat`, `J_stack.mat`, `B_stack.mat`, copies of `main.m`, `experimentEnvOptConf.m`, `PendubotModelConf.m` and `neuralControllerConf.m` as they were at run time, and, for 31 and 37, the nominal rollout `x_stack.mat` / `u_stack.mat`. The `normalization_data.mat` in each was recovered afterwards (README, section 4.5). |

## `config_templates/`

Older generic configs kept off the MATLAB path: a unit-parameter `PendubotModelConf.m` with one uncertain parameter, and a cart-pole `neuralControllerConf.m` (`[6 10 3]`).

## `docs/`

| Path | Content |
|---|---|
| `dissertation_report.pdf` | The MSc dissertation (43 pages), converted from the submitted Word file with LibreOffice. README section 9 describes the conversion. |
| `figures/parameter_sweep_x1.png` | README figure, generated by `tools/reproduce_results.py`. |
| `figures/matlab/*.fig` | Original MATLAB figures. README section 4.3 lists what each one actually shows. |
| `data/x_stack.mat`, `u_stack.mat` | Run 31 nominal rollout, re-tested on 4 Sep 2025 with the 23 Aug normalisation (the dissertation's "Adam training" figure). |
| `configs/Robust_ADAM.txt`, `Robust_FEP.txt` | The `adam_opt` and `fep_opt` blocks of `experimentEnvOptConf.m` as it was for run 37. They are not the settings of the reported Adam (run 31) and FEP (run 10) runs; those are in `results/31/` and `results/10/`. |

## `tools/` (Python, optional)

| File | Purpose |
|---|---|
| `fig_metrics.py` | Reads trajectories out of MATLAB `.fig` files without MATLAB, and computes Ts, %OS, RMS(u) and divergence. |
| `pendubot_sim.py` | NumPy re-simulation of the closed loop. It reproduces MATLAB to about 1e-12. `--plant as-run` gives the swapped parameters of runs 10, 31 and 37; `--plant corrected` gives the current config. Also decodes the `casadi.DM` stored in `results/37/optimal_param.mat`. |
| `reproduce_results.py` | Regenerates the README tables and figure from the run folders. |
| `selftest.py` | Consistency checks run by GitHub Actions (`.github/workflows/verify.yml`). |
