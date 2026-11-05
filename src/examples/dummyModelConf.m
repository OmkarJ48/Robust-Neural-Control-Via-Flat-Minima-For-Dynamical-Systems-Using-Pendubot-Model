%% dummy model for optimizing test function such as himmelblau or others


function [v,x,u,xdot,v_nominal,L,final_L_func,xInit, struct_env_var, absolute_path_to_save] = dummyModelConf()

    %% init and final state
    xInit = [0]; % initial state
    %% cart pole configuration file
    x1 = casadi.SX.sym('x1');
    x  = [x1];

    % im gonna use this directly for the test function inside the cost function
    u1 = casadi.SX.sym('u1');
    u2 = casadi.SX.sym('u2');
    u = [u1];

    v1 = casadi.SX.sym('v1');
    v2 = casadi.SX.sym('v2');
    v = [v1 v2];      %(column vector)



    % nominal value for the uncertain parameters (column vector)
    v_nominal = [-1.829005199391395;1];

    % empty model
    xdot = x;

    k1 = v(1);
    k2 = v(2);
    k3 = u(1);

    %% Test Function
    %z1 =  -(k1.^2 + k2 -11).^2;
    %z2 =  +(k1 + k2.^2 - 7).^2;
    z_tot =  -(k1^2 + k2 + k3 -11)^2  +(k1 + k2^2 + k3 - 7)^2 + (k1+k2+k3^2 - 5)^2;
    L = 0;

    %% empty final term 
    disp('Final term')
    final_L_func = casadi.Function('final_L', {v, x, u}, {z_tot});
    %% visualization paramaters
    struct_env_var.lb   = [-5 -5];
    struct_env_var.ub   = [5 5];
    struct_env_var.step = [0.1 0.1];

     % to visualize multiple resulting behaviours while varying uncertain
    % parameters
    struct_env_var = creatingGridBoundUncertainVariable(v_nominal,struct_env_var);
    
    % Save
    absolute_path_to_save = mfilename('fullpath');


end