
%% add management for collocated system
classdef lqrController < controller.baseController
    
    
    properties
        A_fun
        B_fun
        A_lqr
        B_lqr
        K_lqr
        S_lqr
        Q
        R
        first_control_iteration
    end
    
    
    methods
        % gain_Q = vector of gains for matrix Q
        % gain_R = vector of gains for matrix R
        function obj = lqrController(modelToLinearize,gain_Q,gain_R,useCollocated)
            
            obj.modelToLin = modelToLinearize;
            obj.in_size    = length(modelToLinearize.x);
            obj.out_size   = length(modelToLinearize.u);
            
            A = jacobian(obj.modelToLin.x_dot, obj.modelToLin.x);
            obj.A_fun = casadi.Function('A_fun', {obj.modelToLin.x, obj.modelToLin.u, obj.modelToLin.v}, {A},{'X','U','w0'}, {'A'});
            
            B = jacobian(obj.modelToLin.x_dot, obj.modelToLin.u);
            obj.B_fun = casadi.Function('B_fun', {obj.modelToLin.x, obj.modelToLin.u, obj.modelToLin.v}, {B},{'X','U','w0'}, {'B'});
            
            if(length(gain_Q)>1)
                obj.Q = diag(gain_Q);
            else
                obj.Q = gain_Q*eye(obj.in_size);
            end
            if(length(gain_R)>1)
                obj.R = diag(gain_R); 
            else
                obj.R = gain_R*eye(obj.out_size);
            end
            obj.useCollocated = useCollocated;
            obj.first_control_iteration = true;
                           
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
        
        
        
        %% check u_nominal
        function tau = controller(obj, Xk, xDes, v_nominal)
            if(obj.norm_in)
                Xk = obj.normalize(Xk);
            end
            if(obj.first_control_iteration)
                % build only one parameters set 
                ndofs  = length(xDes)/2;
                q_des  = xDes(1:ndofs);
                qd_des = xDes(ndofs+1:end);
                %% compute u_nominal
                g_eq = obj.modelToLin.gravity(q_des,v_nominal);                    
                u_nominal = g_eq(1);
                
                A_lin = full(obj.A_fun(xDes,u_nominal,v_nominal));
                B_lin = full(obj.B_fun(xDes,u_nominal,v_nominal));
                obj.A_lqr = A_lin;
                obj.B_lqr = B_lin; 
                
                [K,S,~] = lqr(A_lin,B_lin,obj.Q,obj.R,0);
                
                obj.K_lqr = K;
                obj.S_lqr = S;
                obj.first_control_iteration = false;
            end
            
            
            %% to add (eventually) mechanics for collocated 
            tau = obj.K_lqr * (xDes - Xk);
%             
            
            if(obj.denorm_out)
                tau = obj.denormalize(tau);
            end
        end
        
       
        
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
end