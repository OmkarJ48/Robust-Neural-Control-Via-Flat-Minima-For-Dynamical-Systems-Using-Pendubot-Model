classdef Pendubot < model.baseModel
    
    properties 
        m_1
        m_2
        l_1
        l_2
        a_1
        a_2
        mu_1
        mu_2
        %COLLOCATED DYNAMICS
        %GENERAL DYNAMICS
    end
    
    methods
        % v                   vector of symbolic values
        % fixed_parameters    vector of values
        %map_2_assign_value   containers.Map({ 'lpole' 'm_c' 'm_p' }, {fp_1 fp_2 v_1});
        function obj = Pendubot(x,u,v,fixed_parameters,map_2_assign_value)
             obj = obj@model.baseModel(x,u,v,fixed_parameters,map_2_assign_value);
                         
             % Collocated
             obj.desDoFToControl = [1;3];
        end
        
        function xDot(obj)
            obj.x_dot(1) = obj.x(3);
            obj.x_dot(2) = obj.x(4);                        
            obj.x_dot(3:4) = inv(obj.M(obj.x(1:length(obj.x)/2), obj.v)) * (-obj.C(obj.x, obj.v)*obj.x(length(obj.x)/2+1:end) - obj.gravity(obj.x(1:length(obj.x)/2), obj.v) + [obj.u;0] - diag([obj.mu_1, obj.mu_2]) * obj.x(length(obj.x)/2+1:end));
            % Viscous friction acts per joint. Runs 10, 31 and 37 used
            % [mu_1;mu_2]' * q_dot, which subtracted one scalar from both rows.
        end
        
        function createM(obj)
            I_1 = (obj.m_1 * obj.a_1^2) / 12;
            I_2 = (obj.m_2 * obj.a_2^2) / 12;

            A_1 = I_1 +I_2 + obj.m_1 * obj.l_1^2 + obj.m_2 * (obj.a_1^2 + obj.l_2^2);
            A_2 = I_2 + obj.m_2 * obj.l_2^2;
            A_3 = obj.m_2 * obj.a_1 * obj.l_2;
            %A_4 = obj.m_2 * obj.a_1 +  obj.m_1 * obj.l_1;
            %A_5 = obj.m_2 * obj.l_2;    
            obj.M_full = [A_1 + 2 * A_3 * cos(obj.x(2)), A_2 + A_3 * cos(obj.x(2));
                  A_2 + A_3 * cos(obj.x(2)), A_2];
            obj.M_collocated = obj.M_full(1,1) - obj.M_full(1,2) * inv(obj.M_full(2,2)) * obj.M_full(2,1);
            obj.M = casadi.Function('M', {obj.x(1:length(obj.x)/2), obj.v}, {obj.M_full});            
        end
        
        function createC(obj)  
            A_3 = obj.m_2 * obj.a_1 * obj.l_2; 
            obj.C_full = [-A_3 * sin(obj.x(2)) * obj.x(4), -A_3 * sin(obj.x(2)) * (obj.x(3)+obj.x(4));
                   A_3 * sin(obj.x(2)) * obj.x(3),  0]; 
            obj.C_collocated = obj.C_full(1,:) - obj.M_full(1,2) * inv(obj.M_full(2,2)) * obj.C_full(2,:);
            obj.C = casadi.Function('C', {obj.x, obj.v}, {obj.C_full});
        end
        
        function createGravity(obj)
            g  = 9.8;
            A_4 = obj.m_2 * obj.a_1 +  obj.m_1 * obj.l_1;
            A_5 = obj.m_2 * obj.l_2;  
            obj.gravity_full = [A_4 * g * sin(obj.x(1)) + A_5 * sin(obj.x(1) + obj.x(2)) * g; A_5 * g * sin(obj.x(1) + obj.x(2))];
            obj.gravity_collocated = obj.gravity_full(1) - obj.M_full(1,2) * inv(obj.M_full(2,2)) * obj.gravity_full(2,:);
            obj.gravity = casadi.Function('G', {obj.x(1:length(obj.x)/2), obj.v}, {obj.gravity_full});
        end
        
    end
end