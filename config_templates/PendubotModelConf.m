
function [v,x,u,xdot,v_nominal,L,final_L_func,xInit,struct_env_var,pendubot_model,x_des, absolute_path_to_save]=PendubotModelConf()

    %% init and final state
    x_des = [0,;pi;0;0]; % desired final state    (the computation of the error with wrap2pi si valid only if the reference for the pole angle is 0 otherwise you need to rotate wrap2pi)
%     xInit = [0;deg2rad(5);0;0]; % initial state
    xInit = [0;0.6 * pi;0;0];

    %% cart pole configuration file
    x1 = casadi.SX.sym('x1');
    x2 = casadi.SX.sym('x2');
    x3 = casadi.SX.sym('x3');
    x4 = casadi.SX.sym('x4');
    x  = [x1; x2; x3; x4];

    u1 = casadi.SX.sym('u1');
    %u2 = casadi.SX.sym('u2');
    u = u1;

    v1 = casadi.SX.sym('v1');
%     v2 = casadi.SX.sym('v2');
    v = [v1];      %(column vector)


    % Pendubot Model

%     l_1 = 0.1032;
%     l_2 = 0.1065;
%     m_1 = 0.193;
%     m_2 = 0.073;
%     a_1 = 0.1483;
%     a_2 = 0.1804;
    
    l_1 = 1;
    l_2 = 1;
    m_1 = 1;
    m_2 = 1;
    a_1 = 1;
    a_2 = 1;    
    
    
    % nominal value for the uncertain parameters (column vector)
    v_nominal = [m_2];

    fixed_parameters   = [l_1, l_2, m_1, a_1, a_2];
    map_2_assign_value = containers.Map({'l_1','l_2','m_1','m_2','a_1','a_2'}, {'fp_1','fp_2','fp_3','v_1','fp_4','fp_5'});
    pendubot_model    = model.Pendubot(x,u,v,fixed_parameters,map_2_assign_value);

    pendubot_model.xDot();
    xdot = pendubot_model.x_dot;
    
    %% normalization
    x_norm = casadi.SX.sym('x_norm',size(x));
    
    x_norm(2) = 2 * (x(2)-(-pi))/(2*pi-(-pi)) - 1;
    x_norm(4) = 2 * (x(4)-(-20))/(20-(-20)) - 1;
    
    
    x_norm_des(2) = 2 * (x_des(2)-(-pi))/(2*pi-(-pi)) - 1;
    x_norm_des(4) = 0;
    

    %% Objective term
    L = (x(2)-x_des(2))^2 + (x(1)-x_des(1))^2 + 0.001 * u' * u;
%     L = (x(2)-x_des(2))^2 + (x(1)-x_des(1))^2 + 0.1 * (x(4)-x_des(4))^2 + 0.1 * (x(3)-x_des(3))^2;
%     L = (x(2)-x_des(2))^2 + (x(1)-x_des(1))^2 + 0.001 * u' * u + 0.01 * (x(4)-x_des(4))^2 + 0.01 * (x(3)-x_des(3))^2;
%     L = 100 * (x(2)-x_des(2))^2 + u' * u;
%      L = (x_norm_des(2)-x_norm_des(2))^2;

    %% final term 
    disp('Final term')
%     final_L = 100*((x(2)-x_des(2))^2 + (x(1)-x_des(1))^2) + 10 * (x(4)-x_des(4))^2 + 10 * (x(3)-x_des(3))^2;
    final_L = 100*((x(2)-x_des(2))^2 + (x(1)-x_des(1))^2 + (x(4)-x_des(4))^2 + (x(3)-x_des(3))^2);
    final_L_func = casadi.Function('final_L', {v, x, u}, {final_L});

    %% vis parameters
    struct_env_var.lb   = [-0.5];
    struct_env_var.ub   = [0.5];
    struct_env_var.step = [0.01];

    % to visualize multiple resulting behaviours while varying uncertain
    % parameters
    struct_env_var = creatingGridBoundUncertainVariable(v_nominal,struct_env_var);
    
    absolute_path_to_save = mfilename('fullpath');

end

