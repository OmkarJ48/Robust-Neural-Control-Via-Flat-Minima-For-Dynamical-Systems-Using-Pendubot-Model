classdef costAndDerBuilder < handle
    properties
        J               % Total cost (objective function)
        gconstr         % General constraints (currently unused)
        lbg             % Lower bounds for inequality constraints
        ubg             % Upper bounds for inequality constraints
        lb              % Lower bounds for controller parameters
        ub              % Upper bounds for controller parameters
        funJ            % CasADi function to evaluate cost J
        gradJ_Fun       % CasADi function to compute gradient of J
        noise_vec       % Symbolic noise variables for all time steps
        N_step          % Number of simulation steps
    end

    methods
        function obj = costAndDerBuilder(xInit, N, ctrl, intgr, final_cost)
            % Constructor that builds the symbolic cost and gradient

            obj.J = 0;
            obj.gconstr = {};
            obj.lbg = [];
            obj.ubg = [];
            obj.N_step = N;

            % Sanity check: ensure dimensions match
            if intgr.state_dim ~= ctrl.in_size
                error('Controller input size must match integrator state dimension.');
            end
            if intgr.in_dim ~= ctrl.out_size
                error('Controller output size must match integrator input dimension.');
            end

            % Set parameter bounds
            obj.lb = ones(size(ctrl.controller_parameters)) * -10;
            obj.ub = ones(size(ctrl.controller_parameters)) * 10;

            % Initialize symbolic state
            Xk = xInit;

            % Preallocate denormalization matrices if needed
            if ctrl.denorm_out
                n_out = length(ctrl.vec_denorm_out);
                diag_vec = zeros(1, n_out);
                min_vec  = zeros(n_out, 1);

                for i = 1:n_out
                    diag_vec(i) = ctrl.vec_denorm_out{i}.max - ctrl.vec_denorm_out{i}.min;
                    min_vec(i)  = ctrl.vec_denorm_out{i}.min;
                end

                mat_denorm = diag(diag_vec);  % Scaling matrix
            end

            % Preallocate noise vector for all time steps
            obj.noise_vec = casadi.MX.sym('r', intgr.state_dim * N, 1);

            % Main loop to build symbolic cost
            for k = 0:N-1
                % Extract current noise vector slice
                Rk = obj.noise_vec(k * intgr.state_dim + 1 : (k + 1) * intgr.state_dim);

                % Controller output
                Uk = ctrl.ctrl(Xk, ctrl.controller_parameters);

                % Apply output denormalization if required
                if ctrl.denorm_out
                    Uk = 0.5 * mat_denorm * (Uk + ones(size(Uk))) + min_vec;
                end

                % Integrate system forward
                Fk = intgr.F('v0', intgr.env_V, 'x0', Xk, 'r', Rk, 'p', Uk);
                Xk = Fk.xf;

                % Accumulate cost
                if k < N-1
                    obj.J = obj.J + Fk.qf;
                else
                    obj.J = obj.J + final_cost(intgr.env_V, Xk, Uk);
                end

                % Example state constraint (can be modified or removed)
                obj.lbg = [obj.lbg; -0.25];
                obj.ubg = [obj.ubg;  inf];
            end

            % Define CasADi cost function
            obj.funJ = casadi.Function('Jfun', ...
                {obj.noise_vec, intgr.env_V, ctrl.controller_parameters}, ...
                {obj.J}, ...
                {'r0', 'v0', 'w0'}, {'val'});

            % Define CasADi gradient function
            gradJ_ = gradient(obj.J, ctrl.controller_parameters);
            obj.gradJ_Fun = casadi.Function('gradJ_', ...
                {obj.noise_vec, intgr.env_V, ctrl.controller_parameters}, ...
                {gradJ_}, ...
                {'r0', 'v0', 'w0'}, {'grad'});
        end
    end
end
