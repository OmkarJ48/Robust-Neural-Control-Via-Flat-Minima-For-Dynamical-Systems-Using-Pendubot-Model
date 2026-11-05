
function [v,x,u,xdot,v_nominal,L,final_L_func,xInit,struct_env_var,urdf_model,x_des,absolute_path_to_save]=urdf2casadiWrapperModelConf()

    %% init and final state
%     xInit = [0;    0.6013; 0.1346; -1.5656; -0.0896; 0.9814; 0 ;0 ;0; 0; 0; 0]; % initial state
%     x_des = [pi/4 ;0.3;    0.1346; -1; -0.0896; 0.9814; 0 ;0 ;0; 0; 0; 0]; % desired final state    
    
    % All the Kuka DoFs
    xInit = [0;    0.6013; 0.1346; -1.5656; -0.0896; 0.9814; 0; 0; 0; 0; 0; 0; 0; 0]; % initial state
    x_des = [pi/4 ;0.3;    0.1346; -1; -0.0896; 0.9814; 0; 0 ;0 ;0; 0; 0; 0; 0]; % desired final state    
    
    % urdf nominal parameters values
    %m_load   = 0.5;

    % nominal value for the uncertain parameters (column vector)
    v_nominal = 7.1000; %[1.5];   
%     v_nominal = [4.75];   
    
    % urdf file name
    urdf_name = 'kuka_lwr_env';
    urdf_model = model.urdf2casadiWrapper(urdf_name);

    % set casadi variables
    x = urdf_model.x;
    u = urdf_model.u;
    v = urdf_model.v;
    
    %urdf_model.xDot();
    xdot = urdf_model.x_dot;
    

    %% Objective term
    %% normalization test for cost function
    %L = (x(1)-x_des(1))^2 + (x(2)-x_des(2))^2 + (x(3)-x_des(3))^2 + (x(4)-x_des(4))^2 + (x(5)-x_des(5))^2;  
    x_norm = casadi.SX.sym('x_norm',size(x));
    for i=1:length(x)
        x_norm(i) = 2 * (x(i)-(-pi))/(2*pi-(-pi)) - 1;
    end
    for i=1:length(x)
        x_norm_des(i) = 2 * (x_des(i)-(-pi))/(2*pi-(-pi)) - 1;
    end
    L = (x_norm(1)-x_norm_des(1))^2 + (x_norm(2)-x_norm_des(2))^2 + (x_norm(3)-x_norm_des(3))^2 + (x_norm(4)-x_norm_des(4))^2 + (x_norm(5)-x_norm_des(5))^2 + (x_norm(6)-x_norm_des(6))^2;         
    
    %% final term 
    disp('Final term')
    %final_L = 10*(abs(x(1)-x_des(1)) + abs(x(2)-x_des(2))+abs(x(3)-x_des(3)) + abs(x(4)-x_des(4)) + abs(x(5)-x_des(5)));
    %% normalization test for final cost function 
    final_L = 10*((x_norm(1)-x_norm_des(1))^2 + (x_norm(2)-x_norm_des(2))^2+(x_norm(3)-x_norm_des(3))^2 + (x_norm(4)-x_norm_des(4))^2 + (x_norm(5)-x_norm_des(5))^2 + (x_norm(6)-x_norm_des(6))^2);
    final_L_func = casadi.Function('final_L', {v, x, u}, {final_L});

    %% vis parameters
    struct_env_var.lb   = [0];
    struct_env_var.ub   = [1];
    struct_env_var.step = [0.01];

    % to visualize multiple resulting behaviours while varying uncertain
    % parameters
    struct_env_var = creatingGridBoundUncertainVariable(v_nominal,struct_env_var);

    % Save
    absolute_path_to_save = mfilename('fullpath');
end

