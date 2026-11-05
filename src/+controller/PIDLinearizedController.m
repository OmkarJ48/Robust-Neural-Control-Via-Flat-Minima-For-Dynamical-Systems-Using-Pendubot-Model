% neuralController

%% to do group NN variables

classdef PIDLinearizedController < controller.baseController
    
    
    properties
        modelToLin
    end
    
    
    methods
        function obj = PIDLinearizedController(modelToLinearize, Kp, Ki, Kd, useCollocated)
            
            obj.modelToLin = modelToLinearize;
%             xDot = modelToLinearize.xDot();
            obj.in_size    = length(modelToLinearize.x);
            obj.out_size   = length(modelToLinearize.u);
            
            % build only one parameters set 
            obj.controller_parameters = [Kp;Ki;Kd]; 
            obj.useCollocated         = useCollocated;
        end
        
        function setDenormalizeOutput(obj,vec_denorm_out)
            obj.denorm_out     = true; 
            obj.vec_denorm_out = vec_denorm_out;
            if(length(vec_denorm_out) ~= obj.out_size)
                error('check if the normalization_data.mat has been updated for the current robot model. Its dimension is different from the u vector length ')
            end
        end

        function setNormalizeInput(obj,vec_norm_in)
            obj.norm_in     = true; 
            obj.vec_norm_in = vec_norm_in;
        end
        
        function Xk = normalize(obj,Xk)
            for i=1:length(Xk)
                Xk(i) = 2 * (Xk(i)-obj.vec_norm_in{i}.min)/(obj.vec_norm_in{i}.max-obj.vec_norm_in{i}.min) - 1;
            end
        end
        
        function tau = denormalize(obj,tau)
            for i=1:length(tau)
                tau(i) = (tau(i)+1)/2 * (obj.vec_denorm_out{i}.max - obj.vec_denorm_out{i}.min) + obj.vec_denorm_out{i}.min;
            end
        end
        
        function tau = controller(obj, Xk, xDes, v_nominal)
            if(obj.norm_in)
                Xk = obj.normalize(Xk);
            end
            ndofs  = length(Xk)/2;
            cur_q  = Xk(1:ndofs);
            cur_qd = Xk(ndofs+1:end);
            
            
           if(obj.useCollocated)
              
               [cur_qSelect,ndofsSelect,cur_qdSelect,xDesSelect]=obj.setCollocatedForControl(Xk,xDes);
               Kp = obj.controller_parameters(1)*eye(ndofsSelect);
               Ki = obj.controller_parameters(2)*eye(ndofsSelect);
               Kd = obj.controller_parameters(3)*eye(ndofsSelect);
               tau = obj.modelToLin.M(cur_q,v_nominal)*( Kp*(xDesSelect(1:ndofsSelect)-cur_qSelect) + Kd*(xDesSelect(ndofsSelect+1:end)-cur_qdSelect) )+ obj.modelToLin.C(Xk,v_nominal)*cur_qd + obj.modelToLin.gravity(cur_q,v_nominal);
           else
               Kp = obj.controller_parameters(1)*eye(ndofs);
               Ki = obj.controller_parameters(2)*eye(ndofs);
               Kd = obj.controller_parameters(3)*eye(ndofs);
               tau = obj.modelToLin.M(cur_q,v_nominal)*( Kp*(xDes(1:ndofs)-cur_q) + Kd*(xDes(ndofs+1:end)-cur_qd) )+ obj.modelToLin.C(Xk,v_nominal)*cur_qd + obj.modelToLin.gravity(cur_q,v_nominal);
           end
            
            
            if(obj.denorm_out)
                tau = obj.denormalize(tau);
            end
        end
        
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
end