classdef integra < handle
    % Integrator class for single-step integration with perturbation support

    properties
        f           % CasADi dynamics function
        env_dim     % Dimension of environment variable v
        in_dim      % Dimension of control input u
        state_dim   % Dimension of state x
        DT          % Time step
        F           % One-step forward function (CasADi)
        env_V       % Symbolic environment variable (for CasADi input)
    end

    methods
        function obj = integra(DT_, v, x, u, xdot, L)
            % Constructor to create the integrator

            % Define the dynamics function with CasADi
            obj.f = casadi.Function('f', {v, x, u}, {xdot, L});
            
            % Assign dimensions
            obj.env_dim   = length(v);
            obj.in_dim    = length(u);
            obj.state_dim = length(x);
            obj.DT        = DT_;

            % Define CasADi symbols for integration
            X0 = casadi.MX.sym('X0', obj.state_dim);  % Initial state
            U  = casadi.MX.sym('U', obj.in_dim);      % Control input
            R  = casadi.MX.sym('R', obj.state_dim);   % Perturbation/noise
            V  = casadi.MX.sym('V', obj.env_dim);     % Environment variable
            obj.env_V = V;

            % Compute one-step Euler integration
            [x_dot, Q] = obj.f(V, X0, U);
            X_next = X0 + (x_dot + R) * obj.DT;

            % Define the CasADi integration function
            obj.F = casadi.Function('F', {V, X0, R, U}, {X_next, Q}, ...
                                     {'v0', 'x0', 'r', 'p'}, {'xf', 'qf'});
        end
    end
end






