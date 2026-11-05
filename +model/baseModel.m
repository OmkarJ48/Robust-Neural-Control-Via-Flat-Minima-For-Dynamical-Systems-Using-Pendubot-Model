classdef baseModel < handle
    
    
   properties
       x
       u
       v
       x_dot
       desDoFToControl % Index of the state to be controlled in PIDController. If empty, all the state can be controlled
       M
       C
       gravity        % this is used for control
       M_full         % this is used for control
       C_full         % this is used for control
       gravity_full
       M_collocated
       C_collocated
       gravity_collocated
   end
    
    methods (Abstract) 
       xDot(obj)
       createM(obj)
       createC(obj)
       createGravity(obj)
    end
    
    methods 
        function obj = baseModel(x,u,v,fixed_parameters,map_2_assign_value)
            obj.x     = x;
            obj.u     = u;
            obj.v     = v;
            obj       = model.assignValueToObjField(obj,v,fixed_parameters,map_2_assign_value);
            obj       = model.initializeXDot(obj,x);  
            % Functions
            obj.createM();
            obj.createC();
            obj.createGravity();
            % dynamics
            obj.xDot();
            obj.desDoFToControl = [];
        end
        
        function SetCollocatedDynamics(obj)
            % check if the model admit a collocated dynamics
            if(isempty(obj.desDoFToControl))
                error('the current model does not have any collocated dynamics, check the model in use')
            else
                obj.M = obj.M_collocated;
                obj.C = obj.C_collocated;
                obj.gravity = obj.gravity_collocated;
            end
            
        end
        
        function SetFullDynamics(obj)
             obj.M = obj.M_full;
             obj.C = obj.C_full;
             obj.gravity = obj.gravity_full;
        end
        
    end
    
    
    
end