classdef urdf2casadiWrapper < handle
    
    
    properties  
     dof  % degrees of freedom of the current robot
     x
     u
     v
     x_dot
     inverse_dyn_func
     rnea
     M
     C
     gravity
     M_full
     C_full
     gravity_full
     M_collocated
     C_collocated
     gravity_collocated
     desDoFToControl
     grav
     smds_
     q_
     qd_
     qdd_
     tau_
     flagForInverseDynamics
    end
    
    methods
        function obj = urdf2casadiWrapper(urdf_model)
            % this line is necessary for compability with the baseModel
            % class
            current_path = fileparts(which('model.urdf2casadiWrapper'));
            urdf_path = [current_path '/urdf/' urdf_model '.urdf'];
            %Load urdf and convert to SMDS format
            [obj.smds_,~,obj.v] = extractSystemModel(urdf_path);
            
            %Initialize variables
            obj.q_   = casadi.SX.sym('q',[1,obj.smds_.NB])';
            obj.qd_  = casadi.SX.sym('qd',[1,obj.smds_.NB])';
            obj.qdd_ = casadi.SX.sym('qdd', [obj.smds_.NB,1]);
            obj.u    = casadi.SX.sym('u',[1,obj.smds_.NB])';
            
            obj.tau_ = obj.u;
            obj.x = [obj.q_;obj.qd_];
            obj.dof = length(obj.x)/2;
            
            % Initialize the dynamic quantities
            obj.createM();
            obj.createC();
%             obj.inverseDynamics();
            obj.flagForInverseDynamics = false;
            
            obj.xDot();
            obj.desDoFToControl=[];
        end
        
        function xDot(obj)
            % copied as is from urdf2casadi symbolicForwardDynamics() minus
            % the cpp part and the function construction which is done
            % later on in our code
            smds = obj.smds_;
            q    = obj.q_;
            qd   = obj.qd_;
            qdd  = obj.qdd_;
            tau  = obj.tau_;
            
            
            %Gravity
            gravityModulus = 9.80665;
            g              = [0;0;-gravityModulus];
            a_grav         = [0;0;0;g(1);g(2);g(3)];

            %Articulated body algorithm
            for i = 1:smds.NB
                [ XJ, S{i} ] = jcalc( smds.jaxis{i},smds.jtype{i}, q(i) );
                vJ = S{i}*qd(i);
                Xup{i} = XJ * smds.Xtree{i};
                if smds.parent(i) == 0
                    v{i} = vJ;
                    c{i} = zeros(size(a_grav));		
                else
                    v{i} = Xup{i}*v{smds.parent(i)} + vJ;
                    c{i} = crm(v{i}) * vJ;
                end

                IA{i} = smds.I{1,i};
                pA{i} = crf(v{i}) * smds.I{1,i} * v{i};
            end

            for i = smds.NB:-1:1
                U{i} = IA{i} * S{i};
                d{i} = S{i}' * U{i};
                u{i} = tau(i) - S{i}'*pA{i};
                if smds.parent(i) ~= 0
                    Ia = IA{i} - U{i}/d{i}*U{i}';
                    pa = pA{i} + Ia*c{i} + U{i} * u{i}/d{i};
                    IA{smds.parent(i)} = IA{smds.parent(i)} + Xup{i}' * Ia * Xup{i};
                    pA{smds.parent(i)} = pA{smds.parent(i)} + Xup{i}' * pa;
                end
            end

            for i = 1:smds.NB
                if smds.parent(i) == 0
                    a{i} = Xup{i} * -a_grav + c{i};
                else
                    a{i} = Xup{i} * a{smds.parent(i)} + c{i};
                end
                qdd(i,1) = (u{i} - U{i}'*a{i})/d{i};
                a{i} = a{i} + S{i}*qdd(i);
            end
            % derivative of joints positions
            obj.x_dot  = obj.qd_;
            % derivative of velocities
            obj.x_dot = [obj.x_dot; qdd];
            % saving only acceleration for debugging purposed
            obj.qdd_  = qdd;
        end
        
        % this function is employed for teaching by demonstration purpose
        % we need to call it in order to create the inverse dyn function
        function inverseDynamics(obj)
            smds = obj.smds_;
            q    = obj.q_;
            qd   = obj.qd_;
            qdd  = casadi.SX.sym('qdd',[smds.NB,1]);
            tau  = obj.tau_;
            f_ext = casadi.SX.sym('f_ext', 6,1,smds.NB);
            
            %Gravity
            gravityModulus = 9.80665;
            g              = [0;0;-gravityModulus];
            a_grav         = [0;0;0;g(1);g(2);g(3)];
            
            %RNEA
            %Forward recursion
            for i = 1:smds.NB
              [ XJ, S{i} ] = jcalc( smds.jaxis{i}, smds.jtype{i}, q(i) );
              vJ = S{i}*qd(i);
              %% Compute i_X_pi;
              % the overall transformation from the local frame of body parent(i)
              % and the local frame of body i: i_X_pi = X_Ti * X_Ji
              Xup{i} = XJ * smds.Xtree{i};
              if smds.parent(i) == 0
                v{i} = vJ;
                a{i} = Xup{i}*(-a_grav) + S{i}*qdd(i);
              else
                v{i} = Xup{i}*v{smds.parent(i)} + vJ;
                a{i} = Xup{i}*a{smds.parent(i)} + S{i}*qdd(i) + crm(v{i})*vJ;
              end
              f{i} = smds.I{i}*a{i} + crf(v{i})*smds.I{i}*v{i};
            end

            % f_ext:one (6,1) vector per each link excluding the base(considered fixed for now) 
            f = apply_external_forces( smds.parent, Xup, f, f_ext );


            for i = smds.NB:-1:1
              tau(i,1) = S{i}' * f{i};
              if smds.parent(i) ~= 0
                f{smds.parent(i)} = f{smds.parent(i)} + Xup{i}'*f{i};
              end
            end
            % Define the symbolic function and set its input and output in proper order
            % and with proper names
            % Require the linkExternlWrenches to be supplied as a single matrix of size
            % [6,NB]
            F_ext = casadi.SX.sym('f_ext', 6,smds.NB);
            for ii = 1:smds.NB
                F_ext(:,ii)= f_ext{ii};
            end
            inputVarNames = {'state','qdd','F_ext','env_variables'};
            outputVarName = 'jointToques';
            obj.inverse_dyn_func = casadi.Function('rnea',{obj.x,qdd,F_ext,obj.v},{tau},inputVarNames,outputVarName);
            env_var_flag = 1;
        end
        
        function createM(obj)
            smds = obj.smds_;
            q    = obj.q_;
            qd   = obj.qd_;
            qdd  = casadi.SX.sym('qdd',[smds.NB,1]);
            [H_cell,~,~] = computeSymbolicCoriolismatrix(q,qd,qdd,smds);
            H        = cell2mat_casadi(H_cell);
            obj.M = casadi.Function('computeMassMatrix',{q,obj.v},{H},...
                            {'joints_position','env_variables'},...
                            {'massMatrix'});
                        
        end
        
        function createC(obj)
            smds = obj.smds_;
            q    = obj.q_;
            qd   = obj.qd_;
            qdd  = casadi.SX.sym('qdd',[smds.NB,1]);
            [~,~,C_cell] = computeSymbolicCoriolismatrix(q,qd,qdd,smds);
            Coriolis = cell2mat_casadi(C_cell);
            obj.C  = casadi.Function('computeCoriolisMatrix',{obj.x,obj.v},{Coriolis},...
                                {'state','env_variables'},...
                                {'coriolisMatrix'}); 
            obj.C_full  = casadi.Function('computeCoriolisMatrix',{obj.x,obj.v},{Coriolis},...
                                {'state','env_variables'},...
                                {'coriolisMatrix'}); 
        end
            
        function createGravity(obj)
            smds = obj.smds_;
            q    = obj.q_;
            qd = zeros(size(q));
            qdd = 0;
            f_ext = zeros(6,1,smds.NB);
            tau  = obj.tau_;
            
            
            %Gravity
            gravityModulus = 9.80665;
            g              = [0;0;-gravityModulus];
            a_grav         = [0;0;0;g(1);g(2);g(3)];
            
            %RNEA
            %Forward recursion
            for i = 1:smds.NB
              [ XJ, S{i} ] = jcalc( smds.jaxis{i}, smds.jtype{i}, q(i) );
              vJ = S{i}*qd(i);
              %% Compute i_X_pi;
              % the overall transformation from the local frame of body parent(i)
              % and the local frame of body i: i_X_pi = X_Ti * X_Ji
              Xup{i} = XJ * smds.Xtree{i};
              if smds.parent(i) == 0
                v{i} = vJ;
                a{i} = Xup{i}*(-a_grav) + S{i}*qdd(i);
              else
                v{i} = Xup{i}*v{smds.parent(i)} + vJ;
                a{i} = Xup{i}*a{smds.parent(i)} + S{i}*qdd(i) + crm(v{i})*vJ;
              end
              f{i} = smds.I{i}*a{i} + crf(v{i})*smds.I{i}*v{i};
            end

            % f_ext:one (6,1) vector per each link excluding the base(considered fixed for now) 
            f = apply_external_forces( smds.parent, Xup, f, f_ext );


            for i = smds.NB:-1:1
              tau(i,1) = S{i}' * f{i};
              if smds.parent(i) ~= 0
                f{smds.parent(i)} = f{smds.parent(i)} + Xup{i}'*f{i};
              end
            end
            % Define the symbolic function and set its input and output in proper order
            % and with proper names
            % Require the linkExternlWrenches to be supplied as a single matrix of size
            % [6,NB]
            F_ext = casadi.SX.sym('f_ext', 6,smds.NB);
            for ii = 1:smds.NB
                F_ext(:,ii)= f_ext{ii};
            end
            inputVarNames = {'q','env_variables'};
            outputVarName = 'gravity_jointTorques';
            obj.gravity      = casadi.Function('gravity',{q,obj.v},{tau},inputVarNames,outputVarName);
            obj.gravity_full = casadi.Function('gravity',{q,obj.v},{tau},inputVarNames,outputVarName);
            env_var_flag = 1;
        end
        
        %% to check 
        %% adding a method to ex
        function grav = gravityFromInverseDynamics(obj,q,v_nominal)
            
            if (obj.flagForInverseDynamics == false)
                obj.inverseDynamics();
                obj.flagForInverseDynamics = true;
            end
            
            qd = zeros(size(q));
            qdd = 0;
            F_ext = 0;
            
            x_ = [q;qd];
            
            grav = obj.inverse_dyn_func(x_,qdd,F_ext,v_nominal);
            %grav = grav';
        end
        
        
        function SetCollocatedDynamics(obj)
            % check if the model admit a collocated dynamics
            if(isempty(obj.desDoFToControl))
                error('the current model does not have any collocated dynamics, check the model in use')
            else
                obj.M       = obj.M_collocated;
                obj.C       = obj.C_collocated;
                obj.gravity = obj.gravity_collocated;
            end
            
        end
        
        function SetFullDynamics(obj)
             obj.M       = obj.M_full;
             obj.C       = obj.C_full;
             obj.gravity = obj.gravity_full;
        end
        
        
    end
    
    
    
end