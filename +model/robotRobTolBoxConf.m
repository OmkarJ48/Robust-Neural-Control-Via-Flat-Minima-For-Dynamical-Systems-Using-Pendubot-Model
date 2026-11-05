function [v,x,u,xdot,v_nominal,L,final_L_func,xInit,struct_env_var]=robotRobTolBoxConf()

    %% init and final state
    x_des = [deg2rad(-84.4);deg2rad(143.1);deg2rad(13.5);0;0;0;0;0]; % desired final state    (the computation of the error with wrap2pi si valid only if the reference for the pole angle is 0 otherwise you need to rotate wrap2pi)
    xInit = [0;0;0;0;0;0;0;0]; % initial state

    %% cart pole configuration file
    x1 = casadi.SX.sym('x1');
    x2 = casadi.SX.sym('x2');
    x3 = casadi.SX.sym('x3');
    x4 = casadi.SX.sym('x4');
    x5 = casadi.SX.sym('x5');
    x6 = casadi.SX.sym('x6');
    x7 = casadi.SX.sym('x7');
    x8 = casadi.SX.sym('x8');
    x  = [x1; x2; x3; x4; x5; x6; x7; x8];

    u1 = casadi.SX.sym('u1');
    u2 = casadi.SX.sym('u2');
    u3 = casadi.SX.sym('u3');
    u4 = casadi.SX.sym('u4');
    u = [u1;u2;u3;u4];

    v1 = casadi.SX.sym('v1');
%     v2 = casadi.SX.sym('v2');
    v = [v1];      %(column vector)


    % cart-pole model
    m_p = 0.5;

    % nominal value for the uncertain parameters (column vector)
    v_nominal = [m_p];

    fixed_parameters   = [];
    map_2_assign_value = containers.Map({'m_p'}, {'v_1'});
    cart_pole_model    = model.cartPole(x,u,v,fixed_parameters,map_2_assign_value);

    cart_pole_model.xDot();
    xdot = cart_pole_model.x_dot;

    %% Objective term
    L = (x(1)-x_des(1))^2 + (x(2)-x_des(2))^2 + (x(3)-x_des(3))^2 + (x(4)-x_des(4))^2 ;

    %% final term 
    disp('Final term')
    final_L = 10*(abs(x(1)-x_des(1)) +abs(x(2)-x_des(2)) +abs(x(3)-x_des(3)) + abs(x(4)-x_des(4))+ abs(x(5)-x_des(5)) +abs(x(6)-x_des(6)) +abs(x(7)-x_des(7)) + abs(x(8)-x_des(8)));
    final_L_func = casadi.Function('final_L', {v, x, u}, {final_L});

    %% vis parameters
    struct_env_var.lb   = [-0.5];
    struct_env_var.ub   = [0.5];
    struct_env_var.step = [0.01];

end