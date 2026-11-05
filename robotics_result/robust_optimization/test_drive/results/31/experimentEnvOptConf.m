% File for experimental parameters
function out = experimentEnvOptConf()

    %% CONTROL PROBLEM PARAMETERS
    out.N     = 800;      % Number of discrete time steps
    out.DT    = 0.01;     % Integration time step (seconds)

    %% FLATNESS PARAMETERS
    out.epsilon_flat = 10;  % Flatness regularization parameter
    out.dumping      = 10.0; % Damping factor for flatness computation

    out.type_of_flat.order = '2order';
    out.type_of_flat.integral = false;
    out.type_of_flat.sample_int = 10;
    out.type_of_flat.resample_iterations = 5;
    out.type_of_flat.sigma = [0.0005 0.0005];

    %% OPTIMIZATION TYPE
    out.optimization_type = 1;  % Using Adam Projected Gradient

    %% OPTIMIZER-SPECIFIC PARAMETERS

    % --- Adam optimizer ---
    out.adam_opt.opt_iters = 1500;             % Slightly extended optimization
    out.adam_opt.stepSize  = 5e-6;             % Slower learning rate for smooth convergence
    out.adam_opt.beta1     = 0.95;             % Slightly stronger momentum
    out.adam_opt.beta2     = 0.999;            % Standard beta2
    out.adam_opt.batch     = 800;              % Full-batch updates to ensure consistency
    out.adam_opt.iter      = 1;
    out.adam_opt.noise_mu     = [0 0 0 0];     % No mean bias
    out.adam_opt.noise_sigma  = [0.001 0.001 0.001 0.001]; % Controlled noise to encourage flatness

    % --- Yama optimizer ---
    out.yama_opt.opt_iters = 2000;
    out.yama_opt.stepSize  = 0.001;
    out.yama_opt.alpha_F   = 1;
    out.yama_opt.alpha_B   = 10;
    out.yama_opt.J_val_des = 5;
    out.yama_opt.B_val_des = -120;
    out.yama_opt.noise_mu     = [0 0 0 0];
    out.yama_opt.noise_sigma  = [0.001 0.001 0.001 0.001];

    % --- IPOPT optimizer ---
    out.ip_opt.opt_iters = 10000;
    out.ip_opt.J_val_des = 70;
    out.ip_opt.noise_mu     = [0 0 0 0];
    out.ip_opt.noise_sigma  = [0 0 0 0];

    % --- YamaAdaGrad optimizer ---
    out.yamaAda_opt.opt_iters = 1000;
    out.yamaAda_opt.stepSize  = 1e-6;
    out.yamaAda_opt.beta2     = 0.999;
    out.yamaAda_opt.iter      = 1;
    out.yamaAda_opt.alpha_F   = 1000;
    out.yamaAda_opt.alpha_B   = 1;
    out.yamaAda_opt.J_val_des = 10;
    out.yamaAda_opt.B_val_des = -100;
    out.yamaAda_opt.noise_mu     = [0 0 0 0];
    out.yamaAda_opt.noise_sigma  = [0.001 0.001 0.001 0.001];

    % --- FEP optimizer ---
    out.fep_opt.opt_iters = 100;
    out.fep_opt.stepSize  = 0.0001;
    out.fep_opt.iter      = 1;
    out.fep_opt.alpha_F   = 1000;
    out.fep_opt.alpha_B   = 1;
    out.fep_opt.J_val_des = 10;
    out.fep_opt.B_val_des = -100;
    out.fep_opt.noise_mu     = [0 0 0 0];
    out.fep_opt.noise_sigma  = [0 0 0 0];
    out.fep_opt.ineqconstr   = 1;
    out.fep_opt.norm_type    = '2';

    % Save the path to the current config file (used for versioning)
    out.absolute_path_to_save_exp_env_opt_conf = matlab.desktop.editor.getActiveFilename;
end
