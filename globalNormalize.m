function Xk = globalNormalize(Xk, vec_norm_in)
%GLOBALNORMALIZE Normalizes each element of Xk to the range [-1, 1]
%   Applies min-max normalization to scale each element of Xk
%   using its corresponding min/max range in vec_norm_in.
%
% Inputs:
%   Xk          - Numeric vector (e.g., state or control input)
%   vec_norm_in - Cell array of structs with fields:
%                   .min - minimum value for normalization
%                   .max - maximum value for normalization
%
% Output:
%   Xk          - Normalized vector with values in [-1, 1]

    for i = 1:length(Xk)
        min_val = vec_norm_in{i}.min;
        max_val = vec_norm_in{i}.max;

        % Avoid divide-by-zero if range is zero
        range = max_val - min_val;
        if range == 0
            warning('Normalization range is zero at index %d. Setting normalized value to 0.', i);
            Xk(i) = 0;
        else
            Xk(i) = 2 * (Xk(i) - min_val) / range - 1;
        end
    end
end
