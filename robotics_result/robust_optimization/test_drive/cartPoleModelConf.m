
function [v,x,u,xdot,v_nominal,L,final_L_func,xInit,struct_env_var,cart_pole_model,x_des, absolute_path_to_save]=cartPoleModelConf()

    %% init and final state
    x_des = [0,;0;0;0]; % desired final state    (the computation of the error with wrap2pi si valid only if the reference for the pole angle is 0 otherwise you need to rotate wrap2pi)
    xInit = [0;deg2rad(5);0;0]; % initial state

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


    % cart-pole model
    lpole = 0.5;
    m_c   = 1;
    m_p   = 0.5;
    k_c   = 1;
    k_p   = 1;

    % nominal value for the uncertain parameters (column vector)
    v_nominal = [m_p];

    fixed_parameters   = [lpole, m_c];
    map_2_assign_value = containers.Map({'lpole' 'm_c' 'm_p'}, {'fp_1' 'fp_2' 'v_1'});
    cart_pole_model    = model.cartPole(x,u,v,fixed_parameters,map_2_assign_value);

    %cart_pole_model.xDot();
    xdot = cart_pole_model.x_dot;
    
    %% normalization
    x_norm = casadi.SX.sym('x_norm',size(x));
    
    x_norm(2) = 2 * (x(2)-(-pi))/(2*pi-(-pi)) - 1;
    x_norm(4) = 2 * (x(4)-(-20))/(20-(-20)) - 1;
    
    
    x_norm_des(2) = 2 * (x_des(2)-(-pi))/(2*pi-(-pi)) - 1;
    x_norm_des(4) = 0;
    

    %% Objective term
    %L = (x(2)-x_des(2))^2;
     L = (x_norm_des(2)-x_norm_des(2))^2;

    %% final term 
    disp('Final term')
    %final_L = 10*(abs(x(2)-x_des(2)) + abs(x(4)-x_des(4)));
    final_L = 10*((x(2)-x_des(2))^2 + (x(4)-x_des(4))^2);
    final_L = 10*((x_norm(2)-x_norm_des(2))^2 + (x_norm(4)-x_norm_des(4))^2);
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

