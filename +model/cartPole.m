classdef cartPole < model.baseModel
    
    
    properties 
        lpole
        m_c
        m_p
        k_c  
        k_p
    end
    
    methods
        % v                   vector of symbolic values
        % fixed_parameters    vector of values
        %map_2_assign_value   containers.Map({ 'lpole' 'm_c' 'm_p' }, {fp_1 fp_2 v_1});
        function obj = cartPole(x,u,v,fixed_parameters,map_2_assign_value)
             obj = obj@model.baseModel(x,u,v,fixed_parameters,map_2_assign_value);
             
             % Functions
             %obj.createM();
             %obj.createC();
             %obj.createGravity();
             
             % Collocated
             obj.desDoFToControl = [2; 4];
        end
        
        function xDot(obj)
            g = 9.8;
            obj.x_dot(1) = obj.x(3);
            obj.x_dot(2) = obj.x(4);
            obj.x_dot(3) = (-obj.m_p*obj.lpole*sin(obj.x(2))*obj.x(4)^2 + obj.u + obj.m_p*g*cos(obj.x(2))*sin(obj.x(2)))/(obj.m_c + obj.m_p - obj.m_p*cos(obj.x(2))^2);
            obj.x_dot(4) = (-obj.m_p*obj.lpole*cos(obj.x(2))*sin(obj.x(2))*obj.x(4)^2 + obj.u*cos(obj.x(2)) + (obj.m_p + obj.m_c)*g*sin(obj.x(2)))/(obj.lpole*(obj.m_c + obj.m_p - obj.m_p*cos(obj.x(2))^2));
        end
        
        function createM(obj)
            % Collocated M
            obj.M_full = [obj.m_c + obj.m_p -obj.m_p*obj.lpole*cos(obj.x(2)); ...
                  -obj.m_p*obj.lpole*cos(obj.x(2)) obj.m_p*obj.lpole^2];
            obj.M_collocated = obj.M_full(2,2) - obj.M_full(2,1) * inv(obj.M_full(1,1)) * obj.M_full(1,2);
            obj.M = casadi.Function('M', {obj.x(1:length(obj.x)/2), obj.v}, {obj.M_full});
        end
        
        function createC(obj)
            obj.C_full = [0 obj.m_p*obj.lpole*obj.x(4)*sin(obj.x(2)); 0 0];
            obj.C_collocated = obj.C_full(2,:) - obj.M_full(2,1) * inv(obj.M_full(1,1)) * obj.C_full(1,:);
            obj.C = casadi.Function('C', {obj.x, obj.v}, {obj.C_full});
        end
        
        function createGravity(obj)
            g  = 9.8;
            obj.gravity_full = [0; -obj.m_p*g*obj.lpole*sin(obj.x(2))];
            obj.gravity_collocated = obj.gravity_full(2) - obj.M_full(2,1) * inv(obj.M_full(1,1)) * obj.gravity_full(1,:);
            obj.gravity = casadi.Function('gravity', {obj.x(1:length(obj.x)/2), obj.v}, {obj.gravity_full});
        end
        
        function xDot = testXDot(obj, x, v_nominal, tau)
            p  = x(1:length(x)/2);
            dp = x(length(x)/2+1:end);
            xDot = inv(obj.M(p, v_nominal)) * (-obj.C(x, v_nominal)*dp - obj.gravity(p, v_nominal) + tau);
        end
    end
    
    
    
end