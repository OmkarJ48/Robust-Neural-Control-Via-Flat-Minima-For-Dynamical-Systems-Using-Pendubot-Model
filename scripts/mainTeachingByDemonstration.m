% Teaching by Demonstration - Main script

clear variables         % Clear all variables from workspace
close all               % Close all figure windows
clc                     % Clear command window

% Add CasADi symbolic toolbox paths (choose according to your system)
addpath('D:\Family\Omkar\Dissertation - Robust Neural Control via Flatness Optimization for Dynamical Systems\matlab_code\matlab_code\casadi');
import casadi.*         % Import CasADi functions

% Project paths (this script lives in <repo>/scripts; the code is in <repo>/src)
repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo_root, 'src'));
addpath(fullfile(repo_root, 'src', 'examples'));

%% Control flags
flag_mode = 1;          % 0: run teaching and training; 1: test trained controller
save_data = 1;          % 1: enable saving normalization and training data

% Define configuration folder for this experiment
name_folder_conf = 'test_drive';
path_to_conf = fullfile(repo_root, 'robotics_result', 'robust_optimization', name_folder_conf);
addpath(path_to_conf);  % Add config folder to path

teacher_selector = 'lqr';           % Choose expert controller: 'pid' or 'lqr'
data_generation_selector = 'perturbed'; % Choose how to generate training data

% Load experimental configuration
exp_conf = experimentEnvOptConf();

% Extract key config variables
N = exp_conf.N;                     % Time steps for trajectory
DT = exp_conf.DT;                   % Integration step
epsilon_flat = exp_conf.epsilon_flat; 
dumping = exp_conf.dumping;
type_of_flat = exp_conf.type_of_flat;
optimization_type = 0;             % Always set to 0 for TbD
optimization_type_param = exp_conf.adam_opt;

%% Brute force optimizer options
trials = 1;                         % Number of optimizer restarts
multiplier = 0.1;                   % Scaling factor for random init params

%% Expert PID controller gains
Kp = 5.0;
Ki = 0.0;
Kd = 2.5;

%% Expert LQR gains
gain_Q = [5,5,0.5,0.5];
gain_R = 20;

%% Load model (Pendubot is used here)
[v, x, u, xdot, v_nominal, L, final_L_func, xInit, struct_env_var, model_object, xDes] = PendubotModelConf();

%% Perturbed data generation parameters
param_pert.number_of_traj = 20;
param_pert.mu = zeros(size(x));
param_pert.Sigma = 1.0 * eye(length(x));
param_pert.torque_sat = 0.5;

%% Basin data generation parameters
param_basin.mu = zeros(size(x));
param_basin.Sigma = [0,0,0,0;
                     0,0,0,0;
                     0,0,0.1,0;
                     0,0,0,0.5];
rng(10);  % Set seed for reproducibility
param_basin.low = [0.9*pi,-0.3,-0.3,-0.3];  % Lower bounds
param_basin.step = [0.08*pi, 0.1, 0.1, 0.1];
param_basin.up = [1.1*pi, 0.3, 0.3, 0.3];   % Upper bounds
param_basin.torque_sat = 0.5;
param_basin.max_error_allowed = 1;

%% Create integrator object
disp('integra')
integr = integra(DT, v, x, u, xdot, L);

%% If we're in Teaching mode
if flag_mode == 0

    % Instantiate teacher controller (PID or LQR)
    if strcmp(teacher_selector, 'pid')
        teacher = controller.PIDLinearizedController(model_object, Kp, Ki, Kd, true);
    elseif strcmp(teacher_selector, 'lqr')
        teacher = controller.lqrController(model_object, gain_Q, gain_R, false);
    else
        error('Invalid teacher controller selected.');
    end

    % Generate training data based on selected mode
    if strcmp(data_generation_selector, 'perturbed')
        [all_traj_x, all_traj_tau, total_x, total_tau] = perturbedTrajectoriesTbD(xInit, N, teacher, xDes, v_nominal, integr, param_pert);
    elseif strcmp(data_generation_selector, 'basin')
        [all_traj_x, all_traj_tau, total_x, total_tau] = perturbedBasinTbD(xInit, N, teacher, xDes, v_nominal, integr, param_basin);
    else
        error('Invalid data_generation_selector.');
    end

    % Display dataset info
    number_of_dataset_points = length(all_traj_tau);
    number_of_trajectories = number_of_dataset_points / N;
    disp("number of training trajectories="); disp(number_of_trajectories);
    iter_per_epoch = number_of_dataset_points / optimization_type_param.batch;
    disp("number of iterations per epoch="); disp(iter_per_epoch);

    %% Normalize output (torques) across all trajectories
    vec_denorm_out = cell(size(all_traj_tau,1),1);
    for i = 1:size(all_traj_tau,1)
        cur_obj.min = min(all_traj_tau(i,:));
        cur_obj.max = max(all_traj_tau(i,:));
        vec_denorm_out{i} = cur_obj;
    end

    % Apply normalization
    for i = 1:size(all_traj_tau,2)
        all_traj_tau(:,i) = globalNormalize(all_traj_tau(:,i), vec_denorm_out);
    end

    % Save normalization data
    file_name = strcat(path_to_conf,'\normalization_data.mat');
    save(file_name, 'vec_denorm_out');

    %% Build neural controller
    control_method = neuralControllerConf(x, path_to_conf);

    %% Visualize selected training trajectories
    number_of_elements_per_columns = 4;
    number_of_sub_images = length(x);
    subplot_rows = ceil(number_of_sub_images / number_of_elements_per_columns);
    desired_traj = [1 6];  % Select trajectories to visualize

    for ii = desired_traj
        figure
        for i = 1:number_of_sub_images
            subplot(subplot_rows, number_of_elements_per_columns, i);
            grid on; hold on; box on;
            plot(total_x{ii}(i,:));
            xlabel('\# step','interpreter','latex','FontSize',10);
            ylabel('Position','interpreter','latex','FontSize',10);
            legend(['$x_' num2str(i) '$'],'interpreter','latex','FontSize',10);
        end
        sgtitle('teacher controller (computed torque plus pid)');
    end

    %% Optimize neural network parameters using Adam
    disp('costAndDerBuilderTeaching')
    batch_dim = optimization_type_param.batch;
    C_and_deriv = costAndDerBuilderTeaching(batch_dim, integr, control_method);
    optimization_type_param.u_des = all_traj_tau;
    optimization_type_param.x_cur = all_traj_x;

    init_param = rand(size(control_method.controller_parameters)) * multiplier;
    adam = optim.adamOptimizer(control_method, 0, optimization_type_param);
    adam.optimize(C_and_deriv, [], init_param, v_nominal);

    % Save optimized parameters
    file_name = strcat(path_to_conf,'\optimal_param_teaching.mat');
    optimal_param = adam.optimal_param;
    save(file_name, 'optimal_param');

    %% Test trained controller (rollout simulation)
    x_stack_NN = zeros(length(x), N);
    u_stack_NN = zeros(length(u), N);
    Xk = xInit;
    for k = 0:N-1
        R = mvnrnd(zeros(1,integr.state_dim), zeros(1,integr.state_dim), 1);
        Uk = control_method.controller(Xk, adam.optimal_param);
        x_stack_NN(:,k+1) = full(Xk);
        u_stack_NN(:,k+1) = full(Uk);
        Fk = integr.F('r',R,'v0',v_nominal,'x0',Xk,'p',Uk);
        Xk = Fk.xf;
    end

    %% Visualize result of the trained network
    figure
    for i = 1:number_of_sub_images
        subplot(subplot_rows, number_of_elements_per_columns, i);
        grid on; hold on; box on;
        plot(x_stack_NN(i,:)');
        xlabel('\# step','interpreter','latex','FontSize',10);
        ylabel('Position','interpreter','latex','FontSize',10);
        legend(['$x_' num2str(i) '$'],'interpreter','latex','FontSize',10);
    end
    sgtitle('student controller (optimized neural network)');

%% If testing mode is enabled
elseif flag_mode == 1

    % Load and test trained controller
    control_method = neuralControllerConf(x, path_to_conf);
    path_to_training_sol_to_test = strcat(path_to_conf,'\optimal_param_teaching.mat');
    optimal_param_to_test = load(path_to_training_sol_to_test);
    optimal_param_to_test = optimal_param_to_test.optimal_param;

    x_stack_NN = zeros(length(x), N);
    u_stack_NN = zeros(length(u), N);
    Xk = xInit;

    for k = 0:N-1
        R = mvnrnd(zeros(1,integr.state_dim), zeros(1,integr.state_dim), 1);
        Uk = control_method.controller(Xk, optimal_param_to_test);
        x_stack_NN(:,k+1) = full(Xk);
        u_stack_NN(:,k+1) = full(Uk);
        Fk = integr.F('r',R,'v0',v_nominal,'x0',Xk,'p',Uk);
        Xk = Fk.xf;
    end

    % Plot test results
    figure
    number_of_sub_images = length(x);  % Number of states
    number_of_elements_per_columns = 4;
    subplot_rows = ceil(number_of_sub_images / number_of_elements_per_columns);

    for i = 1:number_of_sub_images
        subplot(subplot_rows, number_of_elements_per_columns, i);
        grid on; hold on; box on;
        plot(x_stack_NN(i,:)');
        xlabel('\# step','interpreter','latex','FontSize',10);
        ylabel('Position','interpreter','latex','FontSize',10);
        legend(['$x_' num2str(i) '$'],'interpreter','latex','FontSize',10);
    end
    sgtitle('student controller (optimized neural network)');
end
