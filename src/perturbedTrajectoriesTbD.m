function [all_traj_x, all_traj_tau, all_x, all_tau] = perturbedTrajectoriesTbD(xInit, N, teacher, xDes, v_nominal, integr, param_pert)
% Generates multiple expert trajectories by perturbing initial trajectory
% Inputs:
%   - xInit: Initial state
%   - N: Number of time steps
%   - teacher: Struct or object with a .controller method
%   - xDes: Desired state
%   - v_nominal: Nominal environment input
%   - integr: Integrator object with method F and state_dim
%   - param_pert: Structure with fields:
%       number_of_traj, mu, Sigma, torque_sat
%
% Outputs:
%   - all_traj_x: [state_dim x (N * num_traj)] matrix of all trajectories
%   - all_traj_tau: [input_dim x (N * num_traj)] matrix of all control actions
%   - all_x, all_tau: Cell arrays of individual trajectories

    % Initial trajectory from xInit
  cur_x = xInit;

state_dim = length(xInit);                  % or use integr.state_dim
input_dim = length(full(teacher.controller(xInit, xDes, v_nominal)));  % infer input size

total_tau = zeros(input_dim, N);
total_x   = zeros(state_dim, N);

for i = 1:N
    R = mvnrnd(zeros(1, integr.state_dim), zeros(1, integr.state_dim));
    tau = full(teacher.controller(cur_x, xDes, v_nominal));
    tau = saturateTorque(tau, param_pert.torque_sat);

    total_tau(:, i) = tau;
    total_x(:, i)   = cur_x;

    Fk = integr.F('v0', v_nominal, 'x0', cur_x, 'r', R, 'p', tau);
    cur_x = full(Fk.xf);
end


    % Store the original trajectory
    all_x = cell(param_pert.number_of_traj, 1);
    all_tau = cell(param_pert.number_of_traj, 1);
    all_x{1} = total_x;
    all_tau{1} = total_tau;

    % Generate perturbed variants
    for ii = 2:param_pert.number_of_traj
       cur_xs = zeros(state_dim, N);
       cur_taus = zeros(input_dim, N);

        for jj = 1:N
            x_pert = total_x(:, jj) + mvnrnd(param_pert.mu, param_pert.Sigma)';
            tau = full(teacher.controller(x_pert, xDes, v_nominal));
            tau = saturateTorque(tau, param_pert.torque_sat);  % same limit as the nominal trajectory
            cur_xs(:, jj) = x_pert;
            cur_taus(:, jj) = tau;

        end
        all_x{ii} = cur_xs;
        all_tau{ii} = cur_taus;
    end

    % Combine all into matrix form
    all_traj_x = cat(2, all_x{:});
    all_traj_tau = cat(2, all_tau{:});
end

function tau = saturateTorque(tau, sat_limit)
    tau = max(-sat_limit, min(sat_limit, tau));
end
