classdef baseController < handle
    
    
   properties
       in_size
       out_size
       controller_parameters
       ctrl
       norm_in             % normalization flag input    default false
       denorm_out          % denormalization flag output default false
       vec_denorm_out      % cell array that contains data to denormalize output (is the model is trained on normalized output)
       vec_norm_in         % cell array that contains data to normalize input (if the model is trained on normalized input)
       path_denorm_in_out  % Path for the normalization/denormalization input/output
       
       
       %% properties for teachers
       modelToLin
       useCollocated
   end
    
    methods (Abstract) 
       
    end
    
    methods 
        
        function [cur_qSelect,ndofsSelect,cur_qdSelect,xDesSelect]=setCollocatedForControl(obj,Xk,xDes)
            
            obj.modelToLin.SetCollocatedDynamics();
            
            stateIdx     = obj.modelToLin.desDoFToControl;
            XkSelect     = Xk(stateIdx);
            ndofsSelect  = length(XkSelect)/2;
            cur_qSelect  = XkSelect(1:ndofsSelect);
            cur_qdSelect = XkSelect(ndofsSelect+1:end);
            xDesSelect   = xDes(stateIdx);
        end
       
    end
    
    
    
end