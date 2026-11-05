%% TODO update it with the last controller version (look at neuralcontroller.m)

classdef dummyController < controller.baseController
    
    
    
    methods
        function obj = dummyController(x,controller_dim)
            obj.in_size = length(x);
            x_in      = casadi.MX.sym('X',obj.in_size,1);
            obj.controller_parameters = casadi.MX.sym('U',controller_dim, 1);
            obj.ctrl = casadi.Function('dummy_ctrl',{x_in,obj.controller_parameters},{obj.controller_parameters});
        end
    end
    
    
    
end