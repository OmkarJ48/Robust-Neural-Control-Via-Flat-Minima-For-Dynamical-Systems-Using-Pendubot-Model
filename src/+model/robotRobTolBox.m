classdef robotRobTolBox < model.baseModel
    
    
    properties 
       m_p   
    end
    
    
    
    methods
        % v                   vector of symbolic values
        % fixed_parameters    vector of values
        %map_2_assign_value   containers.Map({ 'lpole' 'm_c' 'm_p' }, {fp_1 fp_2 v_1});
        function obj = robotRobTolBox(x,u,v,fixed_parameters,map_2_assign_value)
             obj = obj@model.baseModel(x,u,v,fixed_parameters,map_2_assign_value);
        end
        
        function xDot(obj)
            g = 9.8;
            obj.x_dot(1) = obj.x(3);
            obj.x_dot(2) = obj.x(4);
            obj.x_dot(3) = (-obj.m_p*obj.lpole*sin(obj.x(2))*obj.x(4)^2 + obj.u + obj.m_p*g*cos(obj.x(2))*sin(obj.x(2)))/(obj.m_c + obj.m_p - obj.m_p*cos(obj.x(2))^2);
            obj.x_dot(4) = (-obj.m_p*obj.lpole*cos(obj.x(2))*sin(obj.x(2))*obj.x(4)^2 + obj.u*cos(obj.x(2)) + (obj.m_p + obj.m_c)*g*sin(obj.x(2)))/(obj.lpole*(obj.m_c + obj.m_p - obj.m_p*cos(obj.x(2))^2));
        end
    end
    
    
    
end