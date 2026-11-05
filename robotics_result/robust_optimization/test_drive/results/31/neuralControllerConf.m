%% neural controller configuration
function [control_method, absolute_path_to_save] = neuralControllerConf(x,path_to_normalization_dat)
    %% neural net parameters
    % Kuka
%     layer_size=[12 6 6];
    % Kuka all DoFs
    %layer_size=[14 6 7];
    % Cart pole
%     layer_size=[4 10 3];
    % Pendubot
    layer_size=[4 10 1];

    % activation function casadi
    blank    = casadi.MX.sym('blank');
    %activation_function = casadi.Function('logistic_f',{blank},{1/(1+exp(-blank))});
    %activation_function = casadi.Function('relu_f',{blank},{max(0,blank)});
    activation_function = casadi.Function('tanh',{blank},{tanh(blank)});
    % last block activation function 
    final_activation_function = casadi.Function('identity',{blank},{blank});
    %final_activation_function = casadi.Function('tanh',{blank},{tanh(blank)});
    %% build controller

    disp('neuralController')
    
    % normalization flag
    out_normalization_flag = 1; % 0 no output normalization, 1 activate output normalization
    in_normalization_flag = 0; % 0 no input normalization, 1 activate input normalization
    
    control_method = controller.neuralController(x, layer_size, activation_function,final_activation_function, out_normalization_flag, in_normalization_flag, path_to_normalization_dat);
    
    % Save
    absolute_path_to_save = mfilename('fullpath');
end