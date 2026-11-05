

classdef adamOptimizer < handle
    
    properties
        epsilon
        stepSize
        beta1
        beta2
        opt_iters
        optimization_type % 0 teaching_by_demonstration, 1 gradient perfomance, 2 Flatness projected in the performance
        optimization_type_param % if 0 optimization_type_param.u_des if 1 [] 
        
        m_
        v_
        m_ext
        v_ext
        param_stack
        cur_grad_stack
        adam_grad_stack
        J_stack 
        optimal_param
        B_stack
        % Debug
        v_stack
        m_stack
    end
    
    methods
        function obj = adamOptimizer(ctrl, optimization_type, optimization_type_param)
            
            % adam parameters
            obj.opt_iters                = optimization_type_param.opt_iters;
            obj.beta1                    = optimization_type_param.beta1;
            obj.beta2                    = optimization_type_param.beta2;
            obj.stepSize                 = optimization_type_param.stepSize;
            obj.epsilon                  = sqrt(eps);
            obj.optimization_type        = optimization_type;
            obj.optimization_type_param  = optimization_type_param;
            
            % - Initialise moment estimates
            obj.m_    = zeros(1, length(ctrl.controller_parameters))';
            obj.v_    = zeros(1, length(ctrl.controller_parameters))';
            obj.m_ext = zeros(1, length(ctrl.controller_parameters))';
            obj.v_ext = zeros(1, length(ctrl.controller_parameters))';
            
            obj.adam_grad_stack  = zeros(obj.opt_iters,length(ctrl.controller_parameters));
            obj.cur_grad_stack   = zeros(obj.opt_iters,length(ctrl.controller_parameters));
            obj.param_stack      = zeros(length(ctrl.controller_parameters),obj.opt_iters);
            obj.J_stack          = zeros(1,obj.opt_iters);
            obj.B_stack          = zeros(1,obj.opt_iters);
            
            obj.v_stack          = zeros(obj.opt_iters,length(ctrl.controller_parameters));
            obj.m_stack          = zeros(obj.opt_iters,length(ctrl.controller_parameters));
        end
        
        function optimize(obj, C_and_deriv, flat, init_parameters, v_nominal)
            
            cur_params = init_parameters;
            J_val      = 0;
            epoch      = 0;
            fig        = figure;
            stop_by_hand = false; % this variable can be used to manually stop the optimization by pausing the code axecution and by setting manually the stop_by_hand = true
            
            for i = 1:obj.opt_iters+1
                
                clf(fig)
                 
                %% optimizing for teaching by demonstration 
                % for teaching we update the gradient for each batch during
                % each epoch
                if(obj.optimization_type == 0)
                    %% this block does not depend from integra.F
                    iter  = obj.optimization_type_param.iter;
                    batch_size = obj.optimization_type_param.batch;
                    % here we shuffle the teaching sequence after each epoch
                    if(iter == 1)
                        shuffle_ind = obj.shuffle(size(obj.optimization_type_param.x_cur,2));
                        obj.optimization_type_param.x_cur   = obj.optimization_type_param.x_cur(:,shuffle_ind);
                        obj.optimization_type_param.u_des   = obj.optimization_type_param.u_des(:,shuffle_ind);
                        %display info cur iter
                        normalization = size(obj.optimization_type_param.x_cur,2)/batch_size;
                        disp([num2str(epoch), ' ', num2str(J_val/normalization)])
                        epoch = epoch + 1;
                        J_val = 0;
                    end
                    % obj.optimization_type_param.x_cur contains the entire
                    % dataset
                    % x_cur contain the current batch
                    
                    % we need to add this check to avoid issue with the
                    % mismatch between the data lenght and the batch size
                    if(iter+batch_size-1>size(obj.optimization_type_param.x_cur,2))
                        cur_end   = size(obj.optimization_type_param.x_cur,2);
                        % since the input dimension of the C_and_deriv.funJ
                        % and C_and_deriv.gradJ_Fun are fixed when the
                        % current iter + batch size > size(obj.optimization_type_param.x_cur,2)
                        % i get exactly the batch size from the end so to
                        % have the right dimension to feed to the J
                        % function and the gradient (even though I have to reuse some data)
                        cur_start = size(obj.optimization_type_param.x_cur,2)-batch_size+1;
                    else
                        cur_end   = iter+batch_size-1;
                        cur_start = iter;
                    end
                    x_cur = obj.optimization_type_param.x_cur(:,cur_start:cur_end);
                    x_cur = x_cur(:);
                    u_des = obj.optimization_type_param.u_des(:,cur_start:cur_end);
                    u_des = u_des(:);
                    
                    obj.optimization_type_param.iter = obj.optimization_type_param.iter + batch_size;
                    
                    % here if with the current batch we get over the data
                    % length we restart and we finish one epoch
                    if obj.optimization_type_param.iter > size(obj.optimization_type_param.x_cur,2)
                        obj.optimization_type_param.iter = 1;
                    end
                    
                    cur_Jval = C_and_deriv.funJ('x_cur',x_cur,'u_des',u_des,'w0',cur_params);
                    cur_Jval = full(cur_Jval.val);
                    J_val    = J_val + cur_Jval;
                    obj.param_stack(:,i) = cur_params;
                    obj.J_stack(i)       = cur_Jval;
                    % computing gradient
                    cur_grad_ = C_and_deriv.gradJ_Fun('x_cur',x_cur,'u_des',u_des,'w0',cur_params);
                    cur_grad_ = full(cur_grad_.grad);
                    
                %% pure perfomance gradient with adam
                elseif(obj.optimization_type == 1)
                    
                    %% generate noise
                    R = mvnrnd(obj.optimization_type_param.noise_mu,obj.optimization_type_param.noise_sigma,C_and_deriv.N_step);
                    R = R';
                    R =R(:);
                    
                    J_val = C_and_deriv.funJ('r0',R,'v0',v_nominal,'w0',cur_params);
                    obj.param_stack(:,i) = cur_params;
                    obj.J_stack(i)       = full(J_val.val);
                   % checking for integral of the flatness around the
                   % nominal parameters
                    if flat.type_of_flat.integral
                        obj.B_stack(i)       = full(flat.B_fun(R,flat.type_of_flat.ball_samples_v,cur_params));
                    else
                        % classic single point flatness
                        obj.B_stack(i)       = full(flat.B_fun(R,v_nominal,cur_params));
                    end
                    
                    %% display info cur iter
                    disp([num2str(i) ' ', num2str(obj.J_stack(i)),' ', num2str(obj.B_stack(i))])
                    
                    %% computing current gradient
                    cur_grad_F = C_and_deriv.gradJ_Fun('r0',R,'v0',v_nominal,'w0',cur_params);
                    cur_grad_F = full(cur_grad_F.grad);
                                     
                    %% pure gradient perfomance
                    cur_grad_ = cur_grad_F;

                    %% plot
%                     subplot(2,1,1);
%                     grid on; hold on;
%                     title('fitness');hold on;
%                     plot(obj.J_stack(1:i));
%                     subplot(2,1,2);
%                     grid on; hold on;
%                     title('flatness');hold on;
%                     plot(obj.B_stack(1:i));
%                     drawnow
              
                end
                
                %% adam optimizer
                nIter = i;
                % - Update biased 1st moment estimate
                obj.m_ = obj.beta1.*obj.m_ + (1 - obj.beta1) .* cur_grad_;
                % - Update biased 2nd raw moment estimate
                obj.v_ = obj.beta2.*obj.v_ + (1 - obj.beta2) .* cur_grad_.^2;
                % - Compute bias-corrected 1st moment estimate
                mHat = obj.m_./(1 - obj.beta1^nIter);
                % - Compute bias-corrected 2nd raw moment estimate
                vHat = obj.v_./(1 - obj.beta2^nIter);
                % - Determine step to take at this iteration
                vfStep_ = obj.stepSize.*mHat./(sqrt(vHat) + obj.epsilon);
                
                %% update parameters
                cur_params = cur_params - vfStep_;         
              
                % stop conditions
                %if(norm(grad_f(i,:) +lambda*grad_B(i,:))<stop_tolerance)
                %    break
                %end
                if(i<=obj.opt_iters)
                    obj.cur_grad_stack(i,:) = cur_grad_';
                    obj.adam_grad_stack(i,:)= vfStep_';
                end
                
                obj.v_stack(i,:)= obj.v_';
                obj.m_stack(i,:)= obj.m_';  

                % this piece of code is only executed when we want to manually stop the optimization for teaching/perfomance 
                % n order to do so it is sufficient to pause the code execution and set manually (in the matlab terminal) the stop_by_hand = true
                if(stop_by_hand)
                    break
                end

            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of optimization

            obj.optimal_param = cur_params;
           
            
        end
        
        
        function  shuffle_ind = shuffle(obj,M_size)
            shuffle_ind =randperm(M_size);  
        end
        
    end
    
    
    
end