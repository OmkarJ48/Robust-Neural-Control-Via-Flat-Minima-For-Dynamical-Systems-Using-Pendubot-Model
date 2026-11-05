
function visualization(N, xInit, v_nominal, x, u, ctrl, adam, integr, C_and_deriv, flat, save_data, data_path, frame_step, struct_env_var)
    
    if(isempty(adam.optimal_param))
        error('the controller parameters are empty fix it! error raised by visualization')
    end
    
    % optimisers are handle objects (isprop); loaded test results are structs (isfield)
    if((isobject(adam) && isprop(adam,'J_stack')) || (isstruct(adam) && isfield(adam,'J_stack')))
        figure;plot(adam.J_stack);title('Cost');
        figure;plot(adam.B_stack);title('Flatness');
    end
    
    % single iteration wih the current env_var
    figure;
    [x_stack,u_stack]=visualizationAux(N,x,u,xInit,v_nominal,adam,ctrl,integr);
    disp('Final state:'), disp(x_stack(end, :));
    if save_data
    save(fullfile(data_path, 'x_stack.mat'), 'x_stack');
    save(fullfile(data_path, 'u_stack.mat'), 'u_stack');
    end


    
    if(save_data)
        path_to_png = strcat(data_path,'\','state_and_control.png');
        %path_to_fig = strcat(data_path,'\','state_and_control.fig');
        saveas(gcf,path_to_png)
        %savefig(gcf,path_to_fig) 
    end
    
    % % multiple iteration with different env var
     sizeinputs = arrayfun(@(low,step,up) low:step:up, struct_env_var.low,struct_env_var.step,struct_env_var.up, 'UniformOutput', false);
     ndarrays = cell(1, length(struct_env_var.low));
     [ndarrays{:}] = ndgrid(sizeinputs{:});
     v_nominal_robust=[];
      for i = 1:length(ndarrays)
     % Precompute total number of combinations
     num_params = length(ndarrays);
     total_combinations = numel(ndarrays{1});

     % Preallocate
     v_nominal_robust = zeros(total_combinations, num_params);

     % Fill each column safely
    for idx = 1:num_params
        v_nominal_robust(:, idx) = ndarrays{idx}(:);  % use 'idx', not 'i'
    end

     end
     
     
     figure
     for ii=1:length(v_nominal_robust)
           
            cur_v_nominal = v_nominal .* (1 + (ii - 1) * 0.1);
            visualizationAux(N, x, u, xInit, cur_v_nominal, adam, ctrl, integr);

     end
    
    %% visualization of the cost function along the environmental variables 2D and 3D
    if(isfield(adam,'param_stack'))
        n_v     = length(v_nominal);
        x_stack = adam.param_stack;
        if(n_v == 1)
            % visualization of enviromental parameters 
            grid_env = struct_env_var.lb(1):struct_env_var.step(1):struct_env_var.ub(1);

            for frame=1:frame_step:length(x_stack)
                figure, hold on
                param_nominal_vec = x_stack(:,frame)*ones(1,length(grid_env));
                z             = C_and_deriv.funJ(grid_env,param_nominal_vec);
                plot(grid_env,full(z));

                 % visualization of the solution over for the current
                 % parameter set 
                z_nominal     = C_and_deriv.funJ(v_nominal,x_stack(:,frame));
                scatter(v_nominal,full(z_nominal),'r')

                string = ['gradient along env vars frame=', num2str(frame), '-- B value = ', num2str(full(flat.B_fun(v_nominal,x_stack(:,frame))))];
                title(string)
                independent_variables = 'v1'; 
                xlabel(independent_variables);
                ylabel('z');
                if(save_data)
                    path_to_fig = strcat(data_path,'\','frame_',num2str(frame),'_2d_v.png');
                    saveas(gcf,path_to_fig)
                end
            end
        elseif(n_v == 2)
            grid_env1 = struct_env_var.lb(1):struct_env_var.step(1):struct_env_var.ub(1);
            grid_env2 = struct_env_var.lb(2):struct_env_var.step(2):struct_env_var.ub(2);
            for frame=1:frame_step:length(x_stack)
                figure, hold on
                [X,Y] = meshgrid(grid_env1,grid_env2);
                Coord_x    = X(:);
                Coord_y    = Y(:);
                param_nominal_vec = x_stack(:,frame)*ones(1,length(Coord_x));
                v_variable  = [Coord_x,Coord_y];

                f  = C_and_deriv.funJ(v_variable',param_nominal_vec);
                %% todo fix reshape in case X != Y
                F  = reshape(full(f),size(X,1),size(X,2));
                surf(X,Y,F)

                independent_variables1 = 'v1'; 
                independent_variables2 = 'v2'; 
                xlabel(independent_variables1);
                ylabel(independent_variables2);
                zlabel('z');
                % plotting red point which corresponds to the current env variables
                % value given the current values of the control variables
                z_nominal     =  C_and_deriv.funJ(v_nominal,x_stack(:,frame));
                scatter3(v_nominal(1),v_nominal(2),full(z_nominal),'r')

                string = ['gradient along env vars frame=', num2str(frame), '-- B value = ', num2str(full(flat.B_fun(v_nominal,x_stack(:,frame))))];
                title(string)

                if(save_data)
                    path_to_fig = strcat(data_path,'\','frame_',num2str(frame),'_3d_x.fig');
                    savefig(gcf,path_to_fig)
                end
            end
        end
    end
    
    
    
    
end