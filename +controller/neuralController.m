% neuralController

%% to do group NN variables

classdef neuralController < controller.baseController
    
    
    properties
        
    end
    
    
    methods
        function obj = neuralController(x, layer_size, activation_function,final_activation_function,...
                                        out_normalization_flag, in_normalization_flag, path_out_normalization)
            %% TODO move this stuff into the base controller constructor and call it here                        
            %% -----------------------------------------------------------------
            obj.denorm_out         = out_normalization_flag;
            obj.norm_in            = in_normalization_flag;
            obj.path_denorm_in_out = path_out_normalization; 
            obj.in_size    = length(x);
            obj.out_size   = layer_size(end);
            
            %% loading data for treating normalized output
            if(obj.denorm_out)
                % check if normalization exist
                path_to_norm_out = strcat(obj.path_denorm_in_out,'\normalization_data.mat');
                if(~isfile(path_to_norm_out))
                    error('the vec_denorm_out.mat is not in the working folder, add it or generate it using teachingbydemonstration or remove the normalization_output_flag in conf')
                end
                cacca = load(path_to_norm_out);
                vec_denorm_out = cacca.vec_denorm_out;
                % check if the normalization is coherent with the current
                % model 
                if(length(vec_denorm_out) ~= obj.out_size)
                    error('check if the vec_denorm_out.mat has been updated for the current robot model. Its dimension is different from the u vector length ')
                end
                obj.vec_denorm_out = vec_denorm_out;
            end
            
            %% loading data for treating normalized input
            if(obj.norm_in)
                 % check if normalization exist
                path_to_norm_in = strcat(obj.path_denorm_in_out,'\normalization_data.mat');
                if(~isfile(path_to_norm_in))
                    error('the vec_norm_in.mat is not in the working folder, add it or generate it using teachingbydemonstration or remove the normalization_output_flag in conf')
                end
                cacca = load(path_to_norm_in);
                vec_norm_in = cacca.vec_norm_in;
                % check if the normalization is coherent with the current
                % model 
                if(length(vec_norm_in) ~= obj.in_size)
                    error('check if the vec_norm_in.mat has been updated for the current robot model. Its dimension is different from the u vector length ')
                end
                obj.vec_norm_in = vec_norm_in;
            end
            %% -----------------------------------------------
            % input nn has to be a column vector
            if(size(x,1)==1)
                input_NN = casadi.MX.sym('input_layer', size(x,2),size(x,1));
            else
                input_NN = casadi.MX.sym('input_layer', size(x,1),size(x,2));
            end
            
            % check input layer dimension vs input size
            if(length(input_NN)~= layer_size(1))
                error('check the number of neurons in layer_size(1).  it has to be equal to length(x) where x is the system state. fix it soaw!');
            end
            
            % building weights and bias
            layers_weigths = cell(length(layer_size)-1,1);
            layers_bias    = cell(length(layer_size)-1,1);
            for i = 1:length(layer_size)-1
                % NN layers
                layer_w_name = strcat('w_',num2str(i));
                layer_b_name = strcat('b_',num2str(i));
                layers_weigths{i} = casadi.MX.sym(layer_w_name,layer_size(i),layer_size(i+1));
                layers_bias{i}    = casadi.MX.sym(layer_b_name,layer_size(i+1),1);
            end 

            %building the controller function by concatenating layers
            % first layer
            input_NN_T = input_NN';
            cur_bias_T = layers_bias{1}';
            output_layer = activation_function((input_NN_T*layers_weigths{1})+cur_bias_T);

            for i = 2:length(layers_weigths)-1
                cur_bias_T = layers_bias{i}';
                output_layer=activation_function((output_layer*layers_weigths{i})+ cur_bias_T);
            end
            % last layer
            cur_bias_T     = layers_bias{end}';
            output_layer   = final_activation_function((output_layer*layers_weigths{end})+cur_bias_T);

            % output_layer   = activation_function((output_layer*layers_weigths{end})+cur_bias_T);
            % output_layer = if_else(output_layer>0.5,10.0,-10.0);

            output_layer_T = output_layer';


            % building the vector of weights and biases
            input_layers_weight  = [];
            input_layers_bias    = [];
            for i = 1: length(layers_weigths)
                input_layers_weight = [input_layers_weight;layers_weigths{i}(:)];
                input_layers_bias    = [input_layers_bias;layers_bias{i}];
            end
            
            % build only one parameters set 
            obj.controller_parameters = [input_layers_weight;input_layers_bias];
            
            obj.ctrl=casadi.Function('neural_network',{input_NN,obj.controller_parameters},{output_layer_T});
            
            
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
        
        function tau = controller(obj, Xk, test_param)
            if(obj.norm_in)
                Xk = obj.normalize(Xk);
            end
           tau = obj.ctrl(Xk, test_param);
           if(obj.denorm_out)
               tau = obj.denormalize(tau);
           end
        end
        
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
end