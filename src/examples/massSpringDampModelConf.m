%% mass spring damp configuration file 

function [v,x,u,xdot,v_nominal,L,final_L_func,xInit,struct_env_var,absolute_path_to_save]=massSpringDampModelConf()

    %% init and final state
    x_des = [2;2;2;2]; % desired final state    (the computatio of the error with wrap2pi si valid only if the reference for the pole angle is 0 otherwise you need to rotate wrap2pi)
    xInit = [1;1;1;1]; % initial state
    %% cart pole configuration file
    x1 = casadi.SX.sym('x1');
    x2 = casadi.SX.sym('x2');
    x3 = casadi.SX.sym('x3');
    x4 = casadi.SX.sym('x4');
    x  = [x1; x2; x3; x4];

    u1 = casadi.SX.sym('u1');
    u2 = casadi.SX.sym('u2');
    u = [u1;u2];

    v1 = casadi.SX.sym('v1');
    %v2 = casadi.SX.sym('v2');
    v = [v1];      %(column vector)


    % massspring_model model

    m_c   = 1;
    m_p   = 0.1;
    k_c   = 1;
    k_p   = 1;

    % nominal value for the uncertain parameters (column vector)
    v_nominal = [m_p];

    fixed_parameters   = [k_p, k_c, m_c];
    map_2_assign_value = containers.Map({ 'k_p' 'k_c' 'm_c' 'm_p' }, {'fp_1' 'fp_2' 'fp_3' 'v_1'});
    cart_pole_model    = model.mobilePointMassSpringDamp(x,u,v,fixed_parameters,map_2_assign_value);

    %cart_pole_model.xDot();
    xdot = cart_pole_model.x_dot;


    %% Objective term
    L = (x(2)-x_des(2))^2;

    %% final term 
    disp('Final term')
    final_L = 100*(abs(x(2)-x_des(2)) + abs(x(4)-x_des(4)));
    final_L_func = casadi.Function('final_L', {v, x, u}, {final_L});

    %% vis parameters
    struct_env_var.lb   = [-1 -1];
    struct_env_var.ub   = [1  1];
    struct_env_var.step = [0.1 0.1];
    
    % to visualize multiple resulting behaviours while varying uncertain
    % parameters
    struct_env_var = creatingGridBoundUncertainVariable(v_nominal,struct_env_var);

    % Save
    absolute_path_to_save = mfilename('fullpath');

end
