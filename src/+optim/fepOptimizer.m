%% got from https://gitlab.com/florian.feppon/null-space-optimizer/-/tree/public-master/nullspace_optimizer
%% TODO we need to add the  
%% Jdes  = obj.optimization_type_param.B_val_des; % value for regulating the flatness (cost function)
%% Cdes  = obj.optimization_type_param.J_val_des; % value for regulating the fitness (constraint function)
%% both to the projected gradient and range gradient and to the merit function as well (TOCHECK)

%% it works with any number of inequalities constraints
classdef fepOptimizer < handle
    
    properties
        
        epsilon
        %beta2
        stepSize
        opt_iters
        optimization_type % 0 teaching_by_demonstration, 1 gradient perfomance, 2 Flatness projected in the performance
        optimization_type_param % optimization_type_param.alpha_F optimization_type_param.J_val_des optimization_type_param.alpha_B
        
        param_stack
        cur_grad_stack
        J_stack 
        optimal_param
        B_stack
        n_constr
        norm_type
        
    end
    
    methods
        function obj = fepOptimizer(ctrl, optimization_type, optimization_type_param)
            
            % yama parameters
            obj.stepSize                 = optimization_type_param.stepSize;
            obj.opt_iters                = optimization_type_param.opt_iters;
            obj.epsilon                  = sqrt(eps);
            obj.optimization_type        = optimization_type;
            obj.optimization_type_param  = optimization_type_param;
            
            
            obj.cur_grad_stack   = zeros(obj.opt_iters,length(ctrl.controller_parameters));
            obj.param_stack      = zeros(length(ctrl.controller_parameters),obj.opt_iters);
            obj.J_stack          = zeros(optimization_type_param.ineqconstr,obj.opt_iters);
            obj.B_stack          = zeros(1,obj.opt_iters);
           
            
            obj.n_constr         = optimization_type_param.ineqconstr;
            obj.norm_type        = optimization_type_param.norm_type;
                
        end
        % C_and_deriv = contains the set of constraints 
            % C_and_deriv.Jval contains the vector value of all constr(column vector)
            % C_and_deriv.gradJ_Fun contains the transpose constraints jacobian (column vector gradient)this is very important!!!!!----------------------------------------------------------------------------------------------------
        % flat = cost functions
            % flat.B_fun = contains the single value cost function
            % flat.gradB_Fun = contains the const function gradient(colum vector) 
        function optimize(obj, C_and_deriv, flat, init_parameters, v_nominal)
            
            cur_params = init_parameters;
            J_val      = 0;
            fig        = figure ;
            
            cur_ortho              = zeros(1,obj.opt_iters);
            cur_grad_fitness_norm  = zeros(1,obj.opt_iters);
            cur_grad_flatness_norm = zeros(1,obj.opt_iters);
            
            alpha = obj.stepSize;
            %% parameters to define in the configuration file
            normalize_tol   = 0;
            tol_merit       = 0;
            itnormalisation = 1;
            K      = 0.1;
            tol_qp = 10^(-20);
            alphaC = 1;
            alphaJ = 1; 
            maxtrials = 100;
            % this has to be moved and maybe it has to be initialized to
            % a number that is different from 0
            normxiJ_save = 0;
            
            % intializing mul to total number of constraints
            mul = zeros(obj.n_constr,1);
            
            % value to control the final value regulation for the fitness
            % and the cost function 
            Jdes  = obj.optimization_type_param.B_val_des; % value for regulating the flatness (cost function)
            Cdes  = obj.optimization_type_param.J_val_des; % value for regulating the fitness (constraint function)
            
            for i = 1:obj.opt_iters+1
                % adding stochastic noise in the process (only when it is specified in the configuration file)
                R = mvnrnd(obj.optimization_type_param.noise_mu,obj.optimization_type_param.noise_sigma,C_and_deriv.N_step);
                R = R';
                R =R(:);
                
                clf(fig)  
                J_val  = C_and_deriv.funJ('r0',R,'v0',v_nominal,'w0',cur_params);
                % managing integral of flatness
                if flat.type_of_flat.integral
                    JB_val = flat.B_fun(R,flat.type_of_flat.ball_samples_v,cur_params);
                else
                    JB_val = flat.B_fun(R,v_nominal,cur_params);
                end
                obj.param_stack(:,i) = cur_params;
                obj.J_stack(:,i)       = full(J_val.val);
                obj.B_stack(i)       = full(JB_val);
                

                %% display info cur iter
                disp([num2str(i),' B=', num2str(obj.B_stack(i)),' C=', num2str(obj.J_stack(:,i)'),' x=',num2str(obj.param_stack(:,i)')])

                %% computing current gradient
                % objective
                if flat.type_of_flat.integral
                    cur_grad_B = flat.gradB_Fun('r0',R,'v0',flat.type_of_flat.ball_samples_v,'w0',cur_params);
                else
                    cur_grad_B = flat.gradB_Fun('r0',R,'v0',v_nominal,'w0',cur_params);
                end
                cur_grad_B = full(cur_grad_B.grad);
                % constraints
                cur_grad_F = C_and_deriv.gradJ_Fun('r0',R,'v0',v_nominal,'w0',cur_params);
                % here we need to transpose it to get the jacobian (since we assume that we are dealing with the matrix of the column gradient of constraints)
                cur_grad_F = full(cur_grad_F.grad)'; % here we get the normal constraints vector jacobian due to the transpose
    
                %% defining the cost jacobian (in our case the flatness index) and auxiliary variables
                J_jac = cur_grad_B;
                prev_mul = mul;
               
                %% solving dual optimization problem
                
                % compute eps for the constraints (perfomance)
                norm_2 = norm(cur_grad_F,2);
                constr_tol = norm_2*alpha*K;
                tilde_tol  = (full(J_val.val) - Cdes) > - constr_tol;  % vector 
                tilde      = (full(J_val.val) - Cdes) >= 0;           % vector
                
                hat = [];
                
                %% tocheck
                % be careful! we did not implemented the check to remove
                % the linear dependent constraints in the constraints
                % jacobian (in the nullspace.py script from the original repo the function that does that is called
                % def eliminate(A))
                
                % if at least one constraints is active 
                if(sum(tilde_tol)>0)
                   % slack variables
                   mu  = casadi.MX.sym('mu', sum(tilde_tol),1);
                   init_parameters = 0;
                   A = cur_grad_F(tilde_tol,:)*cur_grad_F(tilde_tol,:)';
                   q = cur_grad_B'*cur_grad_F(tilde_tol,:)';
                   
                   L = mu'*A*mu + 2*q*mu;
                   g = mu;
                   
                   options = struct;
                   mul_qp  = zeros(size(tilde_tol));
                   %% todo move it in the configuration file 
                   %options.max_iter = 50;
                   % non linear 
                   nlp = struct('x',mu,'f',L,'g',g); 
                   % solver
                   S   = casadi.qpsol('S','qpoases',nlp);
                   mul_qp_res = S('x0',init_parameters,'lbg',zeros(sum(tilde_tol),1),'ubg',inf*ones(sum(tilde_tol),1));
                   mul_qp(tilde_tol) = full(mul_qp_res.x);
                   hat = mul_qp > 30*tol_qp; 
                   % we need to remap the hat vector into the tilde tol vector (it means that hat has to have n_constraints number of element) 
%                    hat_remap = zeros(size(tilde_tol));
%                    hat_remap(tilde_tol) = hat;
%                    hat = hat_remap;
%                    % we need to convert hat as logical vector (true,false)
%                    hat = logical(hat);
                else
                   % we need to remap the hat vector into the tilde tol vector (it means that hat has to have n_constraints number of element) 
                   % we need to do that even in the case when tilde tol is empty
                   hat = zeros(size(tilde_tol));
                end
               
                %% null projection 
                mul = zeros(size(tilde_tol));
                if(sum(hat)==0)
                    % here if hat are all zero ( which it means that or tilde_tol are all zeros or the 
                    % optimization did not find any multiplier different from zero) it means that
                    % the projection has to be an identity so the C matrix
                    % has to be 0 (we di not have any active constraints in the next step)
                    C_jac_hat    = zeros(size(cur_grad_F));
                    DCDCtinv_hat = zeros(size(C_jac_hat,1),size(C_jac_hat,1));
                    mul = - DCDCtinv_hat*C_jac_hat*J_jac;
                    % here we need to define C_jac_hat_T to be consistent with the definition in the else block
                    % even though here it is not really necessary
                    C_jac_hat_T = zeros(size(cur_grad_F));
                    C_jac_hat_T = C_jac_hat_T';
                else
                    C_jac_hat = cur_grad_F(hat,:);
                    DCDCtinv_hat = pinv(C_jac_hat*C_jac_hat'); 
                    mul(tilde_tol) = - DCDCtinv_hat*C_jac_hat*J_jac;
                    % in order to deal with the fact that we need to perform a
                    % matrix multiplication we need to explcitly define the
                    % transpose of C_jac_hat and we need to put to zero all the
                    % colums corresponding to non active constraints w.r.t.
                    % tilde_tol
                    C_jac_hat_T = zeros(size(cur_grad_F));
                    C_jac_hat_T = C_jac_hat_T';
                    C_jac_hat_T(:,hat) = C_jac_hat';
                end
                
                
                %% to check (summing is not gonna work when you have both mul positive and negative)
                index_mul = mul < 0;
                if(sum(index_mul))
                    print("Warning, the active set has not been predicted correctly Using old lagrange multipliers")
                    mul = mul_qp;
                end
                
                
                xiJ = J_jac + C_jac_hat_T*mul;
                
                %% range projection
                % The || operators perform element-wise comparisons between two arrays.
                tilde_star = (tilde + hat);
                tilde_star = tilde_star >0;
                if(sum(tilde_star))
                    C_jac_star = cur_grad_F(tilde_star,:);
                    dCtdCtTinv_star = pinv(C_jac_star*C_jac_star');
                else
                    C_jac_star      = zeros(size(cur_grad_F));
                    dCtdCtTinv_star = zeros(size(C_jac_star,1),size(C_jac_star,1));
                end
                
                J_val_star = full(J_val.val);
                % if neither the multiplier are different from zero nor any
                % costraints is violated in this case we need to zero the
                % range projection so we set j_val_star to a vector of
                % zeros
                if(sum(tilde_star)==0)
                    J_val_star = zeros(size(tilde_star));
                else
                    J_val_star = J_val_star(tilde_star);
                end
                 
                xiC = C_jac_star'*dCtdCtTinv_star*J_val_star;

                
                %% coefficients computation                             
                normxiJ = obj.computeNorm(xiJ);
                mask_mul_tol = mul > normalize_tol;
                mask_prev_mul_tol = prev_mul > normalize_tol;
                flag_mul_tol = true;
                for jj =1:length(mask_mul_tol)
                    if(mask_mul_tol(jj)~=mask_prev_mul_tol(jj))
                        flag_mul_tol = false;
                        break
                    end
                end
                %% TODO CHECK on python if the first iteration it always get into the first if below
                if (i-1) < itnormalisation || (normalize_tol >= 0 && ~(flag_mul_tol))
                    normxiJ_save = normxiJ;
                    AJ = (alphaJ * alpha) / (1e-9 + normxiJ);
                else
                    AJ = (alphaJ * alpha) / max(1e-9 + normxiJ, normxiJ_save);
                end
                AC = min(0.9, (alphaC*alpha) / max(obj.computeNorm(xiC), 1e-9));
                
                
                %% gradient integration with line search
                dx = -AJ*xiJ-AC*xiC;
                % DEBUG
                %disp(normxiJ)
                %disp(normxiJ_save)
                %disp(AC)
                %disp(AJ)
                
                
                success = 0;
                %  int this expression merit = AJ*(J+muls.dot(C))+0.5*AC * C[indicesEps].dot(dCtdCtTinv.dot(C[indicesEps]))
                %  J = full(JB_val)
                %  C = full(J_val.val)

                merit = AJ*(full(JB_val) + mul'*full(J_val.val))+0.5*AC * J_val_star'* dCtdCtTinv_star * J_val_star;
                newx = zeros(size(dx));
                for k=0:maxtrials-1
                     newx = cur_params + (0.5^k)*dx;
                     J_val_new  = C_and_deriv.funJ('r0',R,'v0',v_nominal,'w0',newx);
                     J_val_new_star = full(J_val_new.val);
                     % if neither the multiplier are different from zero nor any
                     % costraints is violated in this case we need to zero the
                     % range projection so we set j_val_star to a vector of
                     % zeros
                     if(sum(tilde_star)==0)
                         J_val_new_star = zeros(size(tilde_star)); 
                     else
                         J_val_new_star = J_val_new_star(tilde_star); 
                     end
                     if flat.type_of_flat.integral
                        JB_val_new = flat.B_fun(R,flat.type_of_flat.ball_samples_v,newx);
                     else
                        JB_val_new = flat.B_fun(R,v_nominal,newx); 
                     end
                     newmerit = AJ*(full(JB_val_new) + mul'*full(J_val_new.val))+0.5*AC * J_val_new_star'* dCtdCtTinv_star * J_val_new_star;
                    if newmerit < (1+sign(merit)*tol_merit)*merit
                        success = 1;
                        break
                    else
                        disp("Warning, merit function did not decrease")
                    end
                end
                if  success == 0
                    disp("All trials have failed, passing to the next iteration.")
                end
                
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
%                 % computation for diagnostic
%                 J1        = cur_grad_F';
%                 J2        = cur_grad_B';
%                 J1pinv    = J1'/(J1*J1');
%                 P1        = eye(length(cur_grad_F)) - J1pinv*J1;
%                 J2P1      = J2*P1;
%                 J2P1pinv  = J2P1'/(J2P1*J2P1');
%                 r1dot     = obj.optimization_type_param.alpha_F*(obj.J_stack(i)-Cdes);
%                 r2dot     = obj.optimization_type_param.alpha_B*(obj.B_stack(i)-Jdes);
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
% 
%                 disp([num2str(cur_ortho(i)),' ',num2str(cur_grad_fitness_norm(i)), ' ',num2str(cur_grad_flatness_norm(i))])
%                         
                
                %% update parameters
                cur_params = newx; 
                % this is used as a stopping condition (which has to be defined)
                normDx = norm(dx);
                
                % stop conditions
                %if(norm(grad_f(i,:) +lambda*grad_B(i,:))<stop_tolerance)
                %    break
                %end
%                 if(i<=obj.opt_iters)
%                     obj.cur_grad_stack(i,:) = cur_grad_';
%                     
%                 end   
%                 
%                 obj.v_stack(i,:)= obj.v_';
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of optimization

            obj.optimal_param = cur_params;
   
        end
        
        function ret=computeNorm(obj,x)
            if(strcmp(obj.norm_type,'inf'))
                ret= norm(x,Inf);
            elseif(strcmp(obj.norm_type,'2'))
                ret= norm(x);
            end
        end
        
    end
    
    
    
end