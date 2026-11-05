% File for experimental parameters
function out = experimentEnvOptConf()

    %% CONTROL PROBLEM PARAMETERS
    out.N     = 800;      % Number of discrete time steps
    out.DT    = 0.01;     % Integration time step (seconds)

    %% FLATNESS PARAMETERS
    % These parameters are used in the flatness-based regularization of the cost function
    out.epsilon_flat = 1e1;  % Flatness regularization parameter (used to compute bounded region)
    out.dumping      = 10.0; % Damping factor for flatness computation

    % Type of flatness used for regularization:
    % - '1orderStd': First-order flatness (based on gradient)
    % - '2order': Second-order flatness (based on Hessian curvature)
    out.type_of_flat.order = '2order';

    % Whether to apply integral flatness using Monte Carlo integration
    out.type_of_flat.integral = false;

    % Number of samples used in Monte Carlo integration for flatness
    out.type_of_flat.sample_int = 10;

    % Max number of resampling attempts if gradient is NaN
    out.type_of_flat.resample_iterations = 5;

    % Standard deviations for sampling environment perturbations
    out.type_of_flat.sigma = [0.0005 0.0005];

    %% OPTIMIZATION TYPE
    % Select optimization method:
    % 0: Teaching by Demonstration (Adam)
    % 1: Performance gradient (Adam)
    % 2: Projected gradient (Yama)
    % 3: IPOPT (interior point optimizer)
    % 4: YamaAdaGrad
    % 5: FEP projected gradient
    out.optimization_type = 5;

    %% OPTIMIZER-SPECIFIC PARAMETERS

    % --- Adam optimizer (used in mode 0 and 1) ---
    out.adam_opt.opt_iters = 1e3;             % Number of optimization iterations
    out.adam_opt.stepSize  = 0.00001;           % Learning rate
    out.adam_opt.beta1     = 0.9;              % Adam momentum term beta1
    out.adam_opt.beta2     = 0.999;            % Adam momentum term beta2
    out.adam_opt.batch     = 500;              % Batch size
    out.adam_opt.iter      = 1;                % (unused) iteration counter
    out.adam_opt.noise_mu     = [0 0 0 0];     % Mean of noise (state dimension)
    out.adam_opt.noise_sigma  = [0 0 0 0];     % Std. dev of noise (state dimension)

    % --- Yama optimizer (used in mode 2) ---
    out.yama_opt.opt_iters = 2e3;
    out.yama_opt.stepSize  = 0.001;
    out.yama_opt.alpha_F   = 1;            % Weight of performance cost
    out.yama_opt.alpha_B   = 1;            % Weight of flatness cost
    out.yama_opt.J_val_des = 7;            % Desired performance cost
    out.yama_opt.B_val_des = -100;         % Desired flatness cost (logarithmic scale)
    out.yama_opt.noise_mu     = [0 0 0 0];
    out.yama_opt.noise_sigma  = [0 0 0 0];

    % --- IPOPT optimizer (used in mode 3) ---
    out.ip_opt.opt_iters = 1e4;
    out.ip_opt.J_val_des = 70;
    out.ip_opt.noise_mu     = [0 0 0 0];
    out.ip_opt.noise_sigma  = [0 0 0 0];

    % --- YamaAdaGrad optimizer (used in mode 4) ---
    out.yamaAda_opt.opt_iters = 1000;
    out.yamaAda_opt.stepSize  = 0.000001;
    out.yamaAda_opt.beta2     = 0.999;
    out.yamaAda_opt.iter      = 1;
    out.yamaAda_opt.alpha_F   = 1000;
    out.yamaAda_opt.alpha_B   = 1;
    out.yamaAda_opt.J_val_des = 10;
    out.yamaAda_opt.B_val_des = -100;
    out.yamaAda_opt.noise_mu     = [0 0 0 0];
    out.yamaAda_opt.noise_sigma  = [0.001 0.001 0.001 0.001];

    % --- FEP optimizer (used in mode 5) ---
    out.fep_opt.opt_iters = 100;
    out.fep_opt.stepSize  = 0.0001;
    out.fep_opt.iter      = 1;
    out.fep_opt.alpha_F   = 1000;
    out.fep_opt.alpha_B   = 1;
    out.fep_opt.J_val_des = 10;
    out.fep_opt.B_val_des = -100;
    out.fep_opt.noise_mu     = [0 0 0 0];
    out.fep_opt.noise_sigma  = [0 0 0 0];
    out.fep_opt.ineqconstr   = 1;   % Enable inequality constraints
    out.fep_opt.norm_type   = '2';   % Enable inequality constraints
    % Save the path to the current config file (used for versioning)
    out.absolute_path_to_save_exp_env_opt_conf = matlab.desktop.editor.getActiveFilename;
end
