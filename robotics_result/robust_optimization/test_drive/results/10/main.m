%% Main Optimization or Testing Script
close all;
clear variables;
clc;

% Add CasADi path
addpath("C:\Family\Omkar\Dissertation - Robust Neural Control via Flatness Optimization for Dynamical Systems\matlab_code\matlab_code\casadi");
import casadi.*

% Configuration folder
name_folder_conf = 'test_drive';
path_to_conf = fullfile(pwd, 'robotics_result', 'robust_optimization', name_folder_conf);
addpath(path_to_conf);

%% Control flags
flag_mode = 1;  % 0: optimize, 1: test existing solution
save_data = 1;
path_save_data = fullfile(path_to_conf, 'results');

%% Load experiment configuration
exp_conf = experimentEnvOptConf();
N = exp_conf.N;
DT = exp_conf.DT;
epsilon_flat = exp_conf.epsilon_flat;
dumping = exp_conf.dumping;
type_of_flat = exp_conf.type_of_flat;
optimization_type = exp_conf.optimization_type;
absolute_path_to_main_file = exp_conf.absolute_path_to_save_exp_env_opt_conf;

%% Initialization method
load_random_predefined_init_point = 'load';  % 'random', 'load', or 'predefined'
fold_numb = 1;
result_from_teaching = 1;
trials = 1;
multiplier = 0.1;

%% Visualization setup
switch optimization_type
    case 1
        opt_iters_for_vis = exp_conf.adam_opt.opt_iters;
    case 2
        opt_iters_for_vis = exp_conf.yama_opt.opt_iters;
    case 3
        opt_iters_for_vis = exp_conf.ip_opt.opt_iters;
    case 4
        opt_iters_for_vis = exp_conf.yamaAda_opt.opt_iters;
    case 5
        opt_iters_for_vis = exp_conf.fep_opt.opt_iters;
end

total_number_of_frame = 10;
frame_step = round(opt_iters_for_vis / total_number_of_frame);

%% Load model (Pendubot)
[v, x, u, xdot, v_nominal, L, final_L_func, xInit, struct_env_var, model_object, xDes, absolute_path_to_save_model] = PendubotModelConf();

%% Initialize integrator and controller
integr = integra(DT, v, x, u, xdot, L);
[control_method, absolute_path_to_save_control] = neuralControllerConf(x, path_to_conf);

%% Cost and flatness builders
C_and_deriv = costAndDerBuilder(xInit, N, control_method, integr, final_L_func);
flat = flatnessBuilder(epsilon_flat, dumping, v_nominal, C_and_deriv, control_method, integr, type_of_flat);

%% Optimization Mode
if flag_mode == 0
    for i = 1:trials
        % Initialize parameters
        switch load_random_predefined_init_point
            case 'random'
                init_param = rand(size(control_method.controller_parameters)) * multiplier;
            case 'load'
                if result_from_teaching
                    path_to_init = fullfile(path_to_conf, 'optimal_param_teaching.mat');
                else
                    path_to_init = fullfile(path_to_conf, 'results', string(fold_numb), 'optimal_param.mat');
                end
                load(path_to_init, 'optimal_param');
                init_param = optimal_param;
            case 'predefined'
                error('Predefined initialization block is very large and should be loaded from external file.');
            otherwise
                error('Invalid initialization method.');
        end

        % Select optimizer
        switch optimization_type
            case {0, 1}
                optimizer = optim.adamOptimizer(control_method, optimization_type, exp_conf.adam_opt);
            case 2
                optimizer = optim.yamaOptimizer(control_method, optimization_type, exp_conf.yama_opt);
            case 3
                optimizer = optim.ipoptOptimizer(control_method, optimization_type, exp_conf.ip_opt);
            case 4
                optimizer = optim.yamaAdaGradOptimizer(control_method, optimization_type, exp_conf.yamaAda_opt);
            case 5
                optimizer = optim.fepOptimizer(control_method, optimization_type, exp_conf.fep_opt);
        end

        optimizer.optimize(C_and_deriv, flat, init_param, v_nominal);

        % Save results
        if save_data
            if ~exist(path_save_data, 'dir'), mkdir(path_save_data); end
            j_folder = 1;
            while exist(fullfile(path_save_data, num2str(j_folder)), 'dir')
                j_folder = j_folder + 1;
            end
            cur_path_save_data = fullfile(path_save_data, num2str(j_folder));
            mkdir(cur_path_save_data);

            % Save optimizer data
            if isprop(optimizer, 'param_stack')
                param_stack = optimizer.param_stack;
                save(fullfile(cur_path_save_data, 'param_stack.mat'), 'param_stack');
            end
            if isprop(optimizer, 'B_stack')
                B_stack = optimizer.B_stack;
                save(fullfile(cur_path_save_data, 'B_stack.mat'), 'B_stack');
            end
            if isprop(optimizer, 'J_stack')
                J_stack = optimizer.J_stack;
                save(fullfile(cur_path_save_data, 'J_stack.mat'), 'J_stack');
            end
            if isprop(optimizer, 'optimal_param')
                optimal_param = optimizer.optimal_param;
                save(fullfile(cur_path_save_data, 'optimal_param.mat'), 'optimal_param');
            end
            save(fullfile(cur_path_save_data, 'init_param.mat'), 'init_param');
            copyfile(absolute_path_to_main_file, cur_path_save_data);
            copyfile(fullfile(path_to_conf, 'experimentEnvOptConf.m'), cur_path_save_data);
            copyfile([absolute_path_to_save_control, '.m'], cur_path_save_data);
            copyfile([absolute_path_to_save_model, '.m'], cur_path_save_data);
        end

        visualization(N, xInit, v_nominal, x, u, control_method, optimizer, integr, C_and_deriv, flat, save_data, cur_path_save_data, frame_step, struct_env_var);
    end

%% Testing Mode
elseif flag_mode == 1
    folder_number = 10;
    switch load_random_predefined_init_point
        case 'load'
            path_to_init = fullfile(path_to_conf, 'results', string(folder_number), 'optimal_param.mat');
            container = load(path_to_init);
        case 'predefined'
            error('Predefined parameters should be loaded externally.');
        otherwise
            error('Invalid mode for testing.');
    end

    visualization(N, xInit, v_nominal, x, u, control_method, container, integr, C_and_deriv, flat, save_data, path_save_data, frame_step, struct_env_var);
else
    error('Invalid flag_mode. Use 0 for optimization, 1 for testing.');
end
