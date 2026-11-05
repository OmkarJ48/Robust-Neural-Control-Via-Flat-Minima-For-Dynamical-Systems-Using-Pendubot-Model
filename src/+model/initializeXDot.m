function obj = initializeXDot(obj,x)
    state_dim = length(x);

    % here we assume for casadi that the casadi variable is sx
    if(isequal(class(x),'casadi.SX'))
        x_dot = [];
        for i = 1:state_dim
            cur_xdot = casadi.SX.sym(['xdot' num2str(i)]);
            x_dot = [x_dot; cur_xdot];
        end
    elseif(isequal(class(x),'sym'))
        x_dot = sym('xdot',[state_dim 1]); 
    elseif(isequal(class(x),'double'))
        x_dot = zeros(state_dim,1);
    else 
        error('missing class value inside model.InitializeXDot')
    end

    obj.x_dot = x_dot;

end