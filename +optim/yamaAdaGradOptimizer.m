

classdef yamaAdaGradOptimizer < handle
    
    properties
        
        epsilon
        beta2
        stepSize
        opt_iters
        optimization_type % 0 teaching_by_demonstration, 1 gradient perfomance, 2 Flatness projected in the performance
        optimization_type_param % optimization_type_param.alpha_F optimization_type_param.J_val_des optimization_type_param.alpha_B
        
        v_
        param_stack
        cur_grad_stack
        J_stack 
        optimal_param
        B_stack
        v_stack
        
    end
    
    methods
        function obj = yamaAdaGradOptimizer(ctrl, optimization_type, optimization_type_param)
            
            % yama parameters
            obj.stepSize                 = optimization_type_param.stepSize;
            obj.opt_iters                = optimization_type_param.opt_iters;
            obj.beta2                    = optimization_type_param.beta2;
            obj.epsilon                  = sqrt(eps);
            obj.optimization_type        = optimization_type;
            obj.optimization_type_param  = optimization_type_param;
            obj.v_    = zeros(1, length(ctrl.controller_parameters))';
            
            
            obj.cur_grad_stack   = zeros(obj.opt_iters,length(ctrl.controller_parameters));
            obj.param_stack      = zeros(length(ctrl.controller_parameters),obj.opt_iters);
            obj.J_stack          = zeros(1,obj.opt_iters);
            obj.B_stack          = zeros(1,obj.opt_iters);
            
            obj.v_stack          = zeros(obj.opt_iters,length(ctrl.controller_parameters));
        end
        
        function optimize(obj, C_and_deriv, flat, init_parameters, v_nominal)
            
            cur_params = init_parameters;
            J_val      = 0;
            fig        = figure ;
            
            cur_ortho              = zeros(1,obj.opt_iters);
            cur_grad_fitness_norm  = zeros(1,obj.opt_iters);
            cur_grad_flatness_norm = zeros(1,obj.opt_iters);
            
            for i = 1:obj.opt_iters+1
                
                R = mvnrnd(obj.optimization_type_param.noise_mu,obj.optimization_type_param.noise_sigma,C_and_deriv.N_step);
                R = R';
                R =R(:);
                
                clf(fig)  
                J_val = C_and_deriv.funJ('r0',R,'v0',v_nominal,'w0',cur_params);
                obj.param_stack(:,i) = cur_params;
                obj.J_stack(i)       = full(J_val.val);
                if flat.type_of_flat.integral
                    obj.B_stack(i)       = full(flat.B_fun(R,flat.type_of_flat.ball_samples_v,cur_params));
                else
                    obj.B_stack(i)       = full(flat.B_fun(R,v_nominal,cur_params));
                end

                %% display info cur iter
                disp([num2str(i) ' ', num2str(obj.J_stack(i)),' ', num2str(obj.B_stack(i))])

                %% computing current gradient
                if flat.type_of_flat.integral
                    cur_grad_B = flat.gradB_Fun('r0',R,'v0',flat.type_of_flat.ball_samples_v,'w0',cur_params);
                else
                    cur_grad_B = flat.gradB_Fun('r0',R,'v0',v_nominal,'w0',cur_params);
                end
                cur_grad_B = full(cur_grad_B.grad);
                cur_grad_F = C_and_deriv.gradJ_Fun('r0',R,'v0',v_nominal,'w0',cur_params);
                cur_grad_F = full(cur_grad_F.grad);

                %% task priority optimization                                
                % version with multitask like kinematic robot (flatness projected in null space of perfomance)
                Bdes      = obj.optimization_type_param.B_val_des;
                J1        = cur_grad_F';
                J2        = cur_grad_B';
                J1pinv    = J1'/(J1*J1');
                P1        = eye(length(cur_grad_F)) - J1pinv*J1;
                J2P1      = J2*P1;
                J2P1pinv  = J2P1'/(J2P1*J2P1');
                r1dot     = obj.optimization_type_param.alpha_F*(obj.J_stack(i)-obj.optimization_type_param.J_val_des);
                r2dot     = obj.optimization_type_param.alpha_B*(obj.B_stack(i)-Bdes);
                cur_grad_ = J1pinv*r1dot + J2P1pinv * (r2dot - J2*J1pinv*r1dot);
                
                %% visualization
%                 subplot(5,1,1);
%                 grid on; hold on;
%                 title('fitness');hold on;
%                 plot(obj.J_stack(1:i));
%                 subplot(5,1,2);
%                 grid on; hold on;
%                 title('flatness');hold on;
%                 plot(obj.B_stack(1:i));
%                 subplot(5,1,3);
%                 grid on; hold on;
%                 title('orthogonality');hold on;
%                 cur_ortho(i) = cur_grad_F' * J2P1pinv * (r2dot - J2*J1pinv*r1dot);
%                 plot(cur_ortho(1:i));
%                 subplot(5,1,4);
%                 grid on; hold on;
%                 title('norm fitness gradient');hold on;
%                 cur_grad_fitness_norm(i) = norm(J1pinv*r1dot);
%                 plot(cur_grad_fitness_norm(1:i));
%                 subplot(5,1,5);
%                 grid on; hold on;
%                 title('norm flatness gradient');hold on;
%                 cur_grad_flatness_norm(i) = norm(J2P1pinv * (r2dot - J2*J1pinv*r1dot));
%                 plot(cur_grad_flatness_norm(1:i));
%                 drawnow

                %disp([num2str(cur_ortho(i)),' ',num2str(cur_grad_fitness_norm(i)), ' ',num2str(cur_grad_flatness_norm(i))])
                        
                %% adagrad optimizer (with exponential decay of the alpha update)
                nIter = i;
                % - Update biased 2nd raw moment estimate
                obj.v_ = obj.beta2.*obj.v_ + (1 - obj.beta2) .* cur_grad_.^2;
                % - Compute bias-corrected 2nd raw moment estimate
                vHat   = obj.v_./(1 - obj.beta2^nIter);
                % compute average vHat
                vHat_avg = (sum(vHat))/length(vHat);
                % - Determine step to take at this iteration
                vfStep_ = obj.stepSize.*cur_grad_./(sqrt(vHat_avg) + obj.epsilon);
                
                %% update parameters
                cur_params = cur_params - vfStep_;    
                
                % stop conditions
                %if(norm(grad_f(i,:) +lambda*grad_B(i,:))<stop_tolerance)
                %    break
                %end
                if(i<=obj.opt_iters)
                    obj.cur_grad_stack(i,:) = cur_grad_';
                    
                end   
                
                obj.v_stack(i,:)= obj.v_';
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of optimization

            obj.optimal_param = cur_params;
   
        end
        
    end
    
    
    
end