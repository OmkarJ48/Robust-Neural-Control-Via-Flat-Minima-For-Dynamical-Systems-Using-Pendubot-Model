% neuralController

%% to do group NN variables

classdef neuralController_old < handle
    
    
    properties
        in_size
        %input_layers_weight
        %input_layers_bias
        controller_parameters
        ctrl
    end
    
    
    methods
        function obj = neuralController_old(x, layer_size, activation_function)
            % input nn has to be a column vector
            if(size(x,1)==1)
                input_NN = casadi.MX.sym('input_layer', size(x,2),size(x,1));
            else
                input_NN = casadi.MX.sym('input_layer', size(x,1),size(x,2));
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
            output_layer   = (output_layer*layers_weigths{end})+cur_bias_T;

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
    
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
end