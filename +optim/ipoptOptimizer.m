

classdef ipoptOptimizer < handle
    
    properties
        
        stepSize
        opt_iters
        optimization_type % 0 teaching_by_demonstration, 1 gradient perfomance, 2 Flatness projected in the performance
        optimization_type_param % optimization_type_param.alpha_F optimization_type_param.J_val_des optimization_type_param.alpha_B
         
        optimal_param
        
        
    end
    
    methods
        function obj = ipoptOptimizer(ctrl, optimization_type, optimization_type_param)
            
            % ipopt parameters
            obj.opt_iters                = optimization_type_param.opt_iters;
            obj.optimization_type        = optimization_type;
            obj.optimization_type_param  = optimization_type_param;
        end
        
        function optimize(obj, C_and_deriv, flat, init_parameters, v_nominal)
            % adding noise to the process
            R = mvnrnd(obj.optimization_type_param.noise_mu,obj.optimization_type_param.noise_sigma,C_and_deriv.N_step);
            R = R';
            R =R(:);


           W0 = casadi.MX.sym('w0', size(init_parameters));
           
           % perfomance
           J_F = C_and_deriv.funJ('r0',R,'v0',v_nominal,'w0',W0);
           % robustness
           if flat.type_of_flat.integral
               %J_B = flat.B_fun('i0',v_nominal,'i1',W0);     
               J_B = flat.B_fun(R,flat.type_of_flat.ball_samples_v,W0);
           else
               J_B = flat.B_fun(R,v_nominal,W0);
           end
           
           disp('Initial Values of losses');
           J_F_0 = C_and_deriv.funJ('r0',R,'v0',v_nominal,'w0',init_parameters);
           if flat.type_of_flat.integral
               %J_B_0 = flat.B_fun('i0',v_nominal,'i1',init_parameters);
               J_B_0 = flat.B_fun(R,flat.type_of_flat.ball_samples_v,init_parameters);
           else
               J_B_0 = flat.B_fun(R,v_nominal,init_parameters);
           end
           disp(J_F_0.val);
           %disp(J_B_0.o0);
           disp(J_B_0);

           % cost function (perfomance + flatness)
           %J = 0.001 * (J_F.val - J_F_0.val) + J_B.o0;
           %J = 0.001 * (J_F.val - obj.optimization_type_param.J_val_des)^2 + J_B.o0;
           J = 0.001 * (J_F.val - obj.optimization_type_param.J_val_des)^2 + J_B;
           %g = J_B.o0 + 20;        
           % options 
           options = struct;
           options.ipopt.max_iter = obj.opt_iters;
           % non linear 
           nlp = struct('x',W0,'f',J);  
           % solver
           S = casadi.nlpsol('S', 'ipopt', nlp,options);
           
           r =S('x0',init_parameters);
                                
           cur_params = full(r.x);
           
           F_final_value =  C_and_deriv.funJ('r0',R,'v0',v_nominal,'w0',cur_params);
           if flat.type_of_flat.integral
               %B_final_value = flat.B_fun('i0',v_nominal,'i1',cur_params);
               B_final_value = flat.B_fun(R,flat.type_of_flat.ball_samples_v,cur_params);
           else
                B_final_value = flat.B_fun(R,v_nominal,cur_params);
           end
           
           disp('Final values');
           disp(F_final_value.val);
           %disp(B_final_value.o0);
           disp(B_final_value);
           
           %save('optimal_params_ipopt','cur_params');
           
           %dlmwrite("optimal_params_ipopt.txt",cur_params,"\t");
         
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end of optimization

           obj.optimal_param = cur_params;
   
        end
        
    end
    
    
    
end