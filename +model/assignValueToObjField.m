function obj = assignValueToObjField(obj,v,fixed_parameters,map_2_assign_value)

    keySet = keys(map_2_assign_value);
            
    for i = 1:length(map_2_assign_value)
        cur_key   = keySet{i};
        [cur_var cur_index] = strtok(map_2_assign_value(cur_key), '_');
        cur_index = str2num(cur_index(2:end));
        if(strcmp(cur_var,'fp'))
            obj.(cur_key) = fixed_parameters(cur_index);
        elseif(strcmp(cur_var,'v'))
            obj.(cur_key) =  v(cur_index);   
        else

        end
    end


end