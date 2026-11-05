%% TODO update it with the last controller version (look at neuralcontrollerconf.m)

%% configuration file dummy controller
function [control_method, absolute_path_to_save] = dummyControllerConf(x)

    controller_dim = 1;
    control_method = controller.dummyController(x, controller_dim);
    
    % Save
    absolute_path_to_save = matlab.desktop.editor.getActiveFilename;
end