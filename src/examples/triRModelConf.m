
function [v,x,u,xdot,v_nominal,L,final_L_func,xInit,struct_env_var,manip3R,x_des, absolute_path_to_save]=triRModelConf()

    %% init and final state
    x_des = [0;0;0;0;0;0]; % desired final state    (the computation of the error with wrap2pi si valid only if the reference for the pole angle is 0 otherwise you need to rotate wrap2pi)
    xInit = [0;0;0;0;0;0]; % initial state

    %% cart pole configuration file
    x1 = casadi.SX.sym('x1');
    x2 = casadi.SX.sym('x2');
    x3 = casadi.SX.sym('x3');
    x4 = casadi.SX.sym('x4');
    x5 = casadi.SX.sym('x5');
    x6 = casadi.SX.sym('x6');
    x  = [x1; x2; x3; x4; x5; x6]; %(column vector)

    u1 = casadi.SX.sym('u1');
    u2 = casadi.SX.sym('u2');
    u3 = casadi.SX.sym('u3');
    u = [u1; u2; u3]; %(column vector)

    v1 = casadi.SX.sym('v1');
%     v2 = casadi.SX.sym('v2');
    v = [v1];      %(column vector)


    % 3r model
    L1_   = 1.5;
    m1_   = 2.9808;
    c1x_  = 0;
    c1y_  = 0;
    c1z_  = 0;
    J1xx_ = 0.5589;
    J1yy_ = 0.3372;
    J1zz_ = 0.2088;
    L2_   = 1.5;
    m2_   = 2.9808;
    c2x_  = 0;
    c2y_  = 0;
    c2z_  = 0;
    J2xx_ = 0.5589;
    J2yy_ = 0.3372;
    J2zz_ = 0.2088;
    L3_   = 1.5;
    m3_   = 2.9808;
    c3x_  = 0;
    c3y_  = 0;
    c3z_  = 0;
    J3xx_ = 0.5589;
    J3yy_ = 0.3372;
    J3zz_ = 0.2088;
   
    % nominal value for the uncertain parameters (column vector)
    v_nominal = [m3_];
    
    fixed_parameters   = [L1_ ,m1_, c1x_, c1y_, c1z_, J1xx_, J1yy_, J1zz_, L2_, m2_, c2x_, c2y_, c2z_, J2xx_, J2yy_, J2zz_, L3_, c3x_, c3y_, c3z_, J3xx_, J3yy_, J3zz_];
    map_2_assign_value = containers.Map({'L1_' 'm1_' 'c1x_' 'c1y_' 'c1z_' 'J1xx_' 'J1yy_' 'J1zz_' 'L2_' 'm2_' 'c2x_' 'c2y_' 'c2z_' 'J2xx_' 'J2yy_' 'J2zz_' 'L3_' 'm3_' 'c3x_' 'c3y_' 'c3z_' 'J3xx_' 'J3yy_' 'J3zz_'}, ...
                                        {'fp_1' 'fp_2' 'fp_3' 'fp_4' 'fp_5' 'fp_6' 'fp_7' 'fp_8' 'fp_9' 'fp_10' 'fp_11' 'fp_12' 'fp_13' 'fp_14' 'fp_15' 'fp_16' 'fp_17' 'v_1' 'fp_18' 'fp_19' 'fp_20' 'fp_21' 'fp_22' 'fp_23'});
    manip3R    = model.triRManipulator(x,u,v,fixed_parameters,map_2_assign_value);
    
    xdot = manip3R.x_dot;
    
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

