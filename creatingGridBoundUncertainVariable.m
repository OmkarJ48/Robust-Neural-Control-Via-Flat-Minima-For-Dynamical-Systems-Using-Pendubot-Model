function struct_env_var = creatingGridBoundUncertainVariable(v_nominal, struct_env_var)
%CREATINGGRIDBOUNDUNCERTAINVARIABLE Generates upper and lower bounds
%   Computes upper and lower bounds for uncertain environment variables 
%   by shifting the nominal vector using specified lower/upper deltas.
%
% Inputs:
%   v_nominal       - 1xN or Nx1 nominal environment parameter vector
%   struct_env_var  - Struct with fields:
%                       lb: lower offsets (same size as v_nominal)
%                       ub: upper offsets (same size as v_nominal)
%
% Output:
%   struct_env_var  - Same struct with new fields:
%                       low: v_nominal + lb
%                       up:  v_nominal + ub

    % Ensure input dimensions match
    if ~isequal(size(v_nominal), size(struct_env_var.lb)) || ...
       ~isequal(size(v_nominal), size(struct_env_var.ub))
        error('Size of v_nominal must match size of lb and ub in struct_env_var.');
    end

    % Compute bounds
    struct_env_var.low = v_nominal + struct_env_var.lb;
    struct_env_var.up  = v_nominal + struct_env_var.ub;
end
