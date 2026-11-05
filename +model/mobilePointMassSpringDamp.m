classdef mobilePointMassSpringDamp < model.baseModel
    
    
    
    properties
        m_c
        m_p
        k_c
        k_p
    end
    
    methods
        % v                   vector of symbolic values
        % fixed_parameters    vector of values
        %map_2_assign_value   containers.Map({ 'lpole' 'm_c' 'm_p' }, {fp_1 fp_2 v_1});
        function obj = mobilePointMassSpringDamp(x,u,v,fixed_parameters,map_2_assign_value)
            obj = obj@model.baseModel(x,u,v,fixed_parameters,map_2_assign_value);
            
            %obj.createM();
            %obj.createC();
            %obj.createGravity();
        end
        
        
        function xDot(obj)
            obj.x_dot(1) = obj.x(3);
            obj.x_dot(2) = obj.x(4);
            obj.x_dot(3) = -obj.m_c*obj.x(3) -obj.k_c*obj.x(1) + obj.u(1);
            obj.x_dot(4) = -obj.m_p*obj.x(4) -obj.k_p*obj.x(2) + obj.u(2);
        end
        
        function createM(obj)
            obj.M    = [1 0; 0 1];
            obj.M_full = [1 0; 0 1];
        end
        
        function createC(obj)
            obj.C    = [obj.m_c 0; 0 obj.m_p];
            obj.C_full = [obj.m_c 0; 0 obj.m_p];
        end
        
        function createGravity(obj)
            obj.gravity = [obj.k_c 0; 0 obj.k_p]*obj.x(1:length(x)/2);
            obj.gravity_full = [obj.k_c 0; 0 obj.k_p]*obj.x(1:length(x)/2);
        end
        
    end
    
    
end