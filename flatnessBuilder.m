%% Flatness Builder Class
% This class computes flatness-based regularization terms (flatness penalties)
% for optimization. It constructs a symbolic function representing the "flatness"
% measure B and its gradient with respect to controller parameters.

classdef flatnessBuilder < handle
    
    properties
        gradB_Fun
        B_fun
        type_of_flat
    end
    
    
    methods
        function obj = flatnessBuilder(epsilon, dumping, v_nominal, C_and_deriv, ctrl, intgr, type_of_flat)
            
            V_multi_numerator   = 1;
            V_multi_denominator = 0;
            delta_J_v           = gradient(C_and_deriv.J, intgr.env_V);
            obj.type_of_flat    = type_of_flat;

            %% todo add a sanity on the selected parameters to verify if they produce nan values or not in that case we resample them  
            if(type_of_flat.integral)
                mu = v_nominal;
                Sigma = diag(type_of_flat.sigma);
                check_dim = size(Sigma);
                if(check_dim(1) ~= length(v_nominal))
                    error('the gaussian covariance matrix for flatness montecarlo int has a different size of v_nominal. check experimentEnvOptConf and current ModelConf!')
                end
                rng('default')  % For reproducibility
                R = mvnrnd(mu,Sigma,type_of_flat.sample_int);
                obj.type_of_flat.ball_samples_v = R(:);
            end

            
            % check if the cost function has a dependency w.r.t. the
            % enviromental variables
            if(numel(delta_J_v)==0)
                error("the current cost function does not depend on the current enviromental variables. The delta_J_v is empty in flatness builder")
            end
            
            if(strcmp(type_of_flat.order,'1orderStd'))
                for i=1:length(intgr.env_V)
                    V_multi_numerator   = V_multi_numerator * sqrt(epsilon)/length(intgr.env_V);
                    V_multi_denominator = V_multi_denominator + (abs(delta_J_v(i)));
                end

                reduced_V_multi_damped = (2^length(intgr.env_V))*(V_multi_numerator/(V_multi_denominator + dumping));
                reduced_B_multi_damped = -log(reduced_V_multi_damped/(2^length(intgr.env_V)));
            elseif(strcmp(type_of_flat.order,'2order'))
                 H_J_v = hessian(C_and_deriv.J,intgr.env_V);
                 curvature = H_J_v.trace;
                 curvature = full(curvature);
                for i=1:length(intgr.env_V)
                    V_multi_numerator   = V_multi_numerator + delta_J_v(i)' * delta_J_v(i);
                end        
                reduced_B_multi_damped = log(0.5 * V_multi_numerator + curvature^2);
            end
            
            obj.B_fun = casadi.Function('B_fun',{C_and_deriv.noise_vec,intgr.env_V,ctrl.controller_parameters}, {reduced_B_multi_damped});
            % integral flatness computed around the nominal
%             if(type_of_flat.integral)
%                v_int_vec = [];
%                 B_int = 0;
%                 v_dim = length(intgr.env_V);
%                 for  k=1:type_of_flat.sample_int
%                     V = [];
%                     for j = 1:v_dim
%                         % single dim
%                         v_cur = casadi.MX.sym(['v' num2str(k) num2str(j)]);
%                         V  = [V;v_cur];
%                         v_int_vec = [v_int_vec(:); v_cur];
%                     end
%                     B_int = B_int + obj.B_fun(C_and_deriv.noise_vec,V,ctrl.controller_parameters);
%                 end
%                 B_int = v_dim*(B_int/type_of_flat.sample_int);
%                 obj.B_fun = casadi.Function('B_fun',{C_and_deriv.noise_vec,v_int_vec,ctrl.controller_parameters}, {B_int});
%                 gradB_ = gradient(B_int, ctrl.controller_parameters);
%                 obj.gradB_Fun = casadi.Function('gradBFun', {C_and_deriv.noise_vec, v_int_vec, ctrl.controller_parameters}, {gradB_}, {'r0','v0','w0'}, {'grad'});
%             else
            % method that computes the flatness around the nominal
            % values 
            gradB_ = gradient(reduced_B_multi_damped, ctrl.controller_parameters);
            obj.gradB_Fun = casadi.Function('gradBFun', {C_and_deriv.noise_vec,intgr.env_V, ctrl.controller_parameters}, {gradB_}, {'r0','v0','w0'}, {'grad'});
%            end
        end


        function [int_grad_B, int_B] = computeIntegralFlat(obj,R,cur_params,v_nominal)
             v_dim = length(v_nominal);
             index = 1;
             valid_sample = 1;
             % this is the center of the ball
             int_grad_B = full(obj.gradB_Fun('r0',R,'v0',v_nominal,'w0',cur_params));
             int_B      = full(obj.B_fun(R,v_nominal,cur_params));
             for  k=1:obj.type_of_flat.sample_int
                    cur_v = obj.type_of_flat.ball_samples_v(index:index+v_dim);
                    cur_grad_B =  full(obj.gradB_Fun('r0',R,'v0',cur_v,'w0',cur_params));
                    cur_B      =  full(obj.B_fun(R,cur_v,cur_params));
                    % here I check if the current gradient computed in this
                    % point (v dimensions) is nan
                    if(isnan(cur_grad_B))
                        disp("one parameter sample is nan")
                        % here we resample a new point
                        ii=1;
                        % we repeat it for a number of times until we fail 
                        while(ii<=obj.type_of_flat.resample_iterations)
                            mu = v_nominal;
                            Sigma = diag(obj.type_of_flat.sigma);
                            cur_v = mvnrnd(mu,Sigma,1);
                            cur_grad_B =  full(obj.gradB_Fun('r0',R,'v0',cur_v,'w0',cur_params));
                            cur_B      =  full(obj.B_fun(R,cur_v,cur_params));
                            if(~isnan(cur_grad_B))
                                valid_sample = valid_sample + 1;
                                int_grad_B = int_grad_B + cur_grad_B;
                                int_B      = int_B + cur_B;
                                break
                            end
                        end
                     else
                         valid_sample = valid_sample + 1;
                         int_grad_B = int_grad_B + cur_grad_B;
                         int_B      = int_B + cur_B;
                     end
                     % udpate the index to get the next point in the ball
                     index = index + v_dim;
             end
             
             int_grad_B = v_dim*(int_grad_B/valid_sample);
             int_B      = v_dim*(int_B/valid_sample);
        end

    end
    



end 
