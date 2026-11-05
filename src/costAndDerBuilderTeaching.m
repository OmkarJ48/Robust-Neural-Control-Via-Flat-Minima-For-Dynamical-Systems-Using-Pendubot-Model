classdef costAndDerBuilderTeaching < handle
    % Cost and Gradient Builder for Teaching by Demonstration (TbD)
    % Builds symbolic CasADi expressions for MSE cost and its gradient
    % using demonstration data and a neural controller
    
    properties
        J               % Symbolic cost function
        Udes            % Desired control input symbols (from teacher)
        Xcur            % Current state symbols
        gconstr         % Placeholder for constraints (not used)
        lbg             % Lower bounds for constraints
        ubg             % Upper bounds for constraints
        lb              % Lower bounds on controller parameters
        ub              % Upper bounds on controller parameters
        funJ            % CasADi function for computing cost
        gradJ_Fun       % CasADi function for computing cost gradient
    end

    methods
        function obj = costAndDerBuilderTeaching(N, intgr, ctrl)
            % Constructor: builds cost and gradient expressions

            % Initialize cost and constraints
            obj.J = 0;
            obj.Xcur = [];
            obj.Udes = [];
            obj.gconstr = {};
            obj.lbg = [];
            obj.ubg = [];

            % Dimension checks
            if intgr.state_dim ~= ctrl.in_size
                error('Controller input size must match state dimension.');
            end
            if intgr.in_dim ~= ctrl.out_size
                error('Controller output size must match action dimension.');
            end

            % Parameter bounds
            obj.lb = -10 * ones(size(ctrl.controller_parameters));
            obj.ub =  10 * ones(size(ctrl.controller_parameters));

            % Build symbolic cost over N steps
            for k = 0:N-1
                % Define symbolic variables for input and output
                Xk = casadi.MX.sym(['x_cur' num2str(k)], ctrl.in_size, 1);
                Uk = ctrl.ctrl(Xk, ctrl.controller_parameters);
                Udes_k = casadi.MX.sym(['u_des' num2str(k)], size(Uk));

                % Accumulate symbols
                obj.Xcur = [obj.Xcur; Xk];
                obj.Udes = [obj.Udes; Udes_k];

                % Accumulate squared error cost
                diff = Udes_k - Uk;
                obj.J = obj.J + sum1(diff.^2);

                % (Optional) add inequality constraints if needed
                obj.lbg = [obj.lbg; -0.25];
                obj.ubg = [obj.ubg; inf];
            end

            % Normalize the cost
            obj.J = obj.J / N;

            % Build CasADi functions
            obj.funJ = casadi.Function('Jfun', ...
                {obj.Xcur, obj.Udes, ctrl.controller_parameters}, ...
                {obj.J}, ...
                {'x_cur', 'u_des', 'w0'}, {'val'});

            gradJ = gradient(obj.J, ctrl.controller_parameters);
            obj.gradJ_Fun = casadi.Function('gradJ_', ...
                {obj.Xcur, obj.Udes, ctrl.controller_parameters}, ...
                {gradJ}, ...
                {'x_cur', 'u_des', 'w0'}, {'grad'});
        end
    end
end
