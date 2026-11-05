classdef triRManipulator < model.baseModel
    
    
    properties 
        L1_
        m1_ 
        c1x_
        c1y_
        c1z_
        J1xx_
        J1yy_
        J1zz_
        L2_
        m2_ 
        c2x_
        c2y_
        c2z_
        J2xx_
        J2yy_
        J2zz_
        L3_
        m3_ 
        c3x_
        c3y_
        c3z_
        J3xx_
        J3yy_
        J3zz_
    end
    
    methods
        % v                   vector of symbolic values
        % fixed_parameters    vector of values
        %map_2_assign_value   containers.Map({ 'lpole' 'm_c' 'm_p' }, {fp_1 fp_2 v_1});
        function obj = triRManipulator(x,u,v,fixed_parameters,map_2_assign_value)
             obj = obj@model.baseModel(x,u,v,fixed_parameters,map_2_assign_value);
             % fully feedback linearizable
             obj.desDoFToControl = [];
        end
        
        function xDot(obj)
             obj.x_dot(1)   = obj.x(4);
             obj.x_dot(2)   = obj.x(5);
             obj.x_dot(3)   = obj.x(6);
             obj.x_dot(4:6) = inv(obj.M(obj.x(1:3), obj.v)) * (-obj.C(obj.x, obj.v) - obj.gravity(obj.x(1:3), obj.v) + obj.u);
        end
        
        function createM(obj)
            %    M_ = MYFILE(J1ZZ,J2ZZ,J3ZZ,L1,L2,L3,C1X,C2X,C3X,C1Y,C2Y,C3Y,M1,M2,M3,Q2,Q3)
            q1   = obj.x(1);
            q2   = obj.x(2);
            q3   = obj.x(3);
            J1zz = obj.J1zz_;
            J2zz = obj.J2zz_;
            J3zz = obj.J3zz_;
            L1   = obj.L1_;
            L2   = obj.L2_;
            L3   = obj.L3_;
            c1x  = obj.c1x_;
            c2x  = obj.c2x_;
            c3x  = obj.c3x_;
            c1y  = obj.c1y_;
            c2y  = obj.c2y_;
            c3y  = obj.c3y_;
            m1   = obj.m1_;
            m2   = obj.m2_;
            m3   = obj.m3_;
            
            t2 = cos(q2);
            t3 = cos(q3);
            t4 = sin(q2);
            t5 = sin(q3);
            t6 = q2+q3;
            t7 = L1.^2;
            t8 = L2.^2;
            t9 = L3.^2;
            t10 = c2x.^2;
            t11 = c3x.^2;
            t12 = c2y.^2;
            t13 = c3y.^2;
            t19 = L2.*c2x.*m2.*2.0;
            t20 = L3.*c3x.*m3.*2.0;
            t14 = cos(t6);
            t15 = sin(t6);
            t16 = m2.*t8;
            t17 = m3.*t8;
            t18 = m3.*t9;
            t21 = m2.*t10;
            t22 = m3.*t11;
            t23 = m2.*t12;
            t24 = m3.*t13;
            t25 = L1.*L2.*m2.*t2;
            t26 = L1.*L2.*m3.*t2;
            t27 = L2.*L3.*m3.*t3;
            t28 = L1.*c2x.*m2.*t2;
            t29 = L2.*c3x.*m3.*t3;
            t30 = L1.*c2y.*m2.*t4;
            t31 = L2.*c3y.*m3.*t5;
            t32 = t27.*2.0;
            t33 = t29.*2.0;
            t34 = t31.*2.0;
            t35 = L1.*L3.*m3.*t14;
            t36 = L1.*c3x.*m3.*t14;
            t37 = L1.*c3y.*m3.*t15;
            t38 = -t30;
            t39 = -t31;
            t40 = -t34;
            t41 = -t37;
            t42 = J3zz+t18+t20+t22+t24+t27+t29+t39;
            t43 = t35+t36+t41+t42;
            t44 = J2zz+J3zz+t16+t17+t18+t19+t20+t21+t22+t23+t24+t25+t26+t28+t32+t33+t35+t36+t38+t40+t41;
            M_ = reshape([J1zz+J2zz+J3zz+t16+t17+t18+t19+t20+t21+t22+t23+t24+t25.*2.0+t26.*2.0+t28.*2.0-t30.*2.0+t32+t33+t35.*2.0+t36.*2.0-t37.*2.0+t40+m1.*t7+m2.*t7+m3.*t7+c1x.^2.*m1+c1y.^2.*m1+L1.*c1x.*m1.*2.0,t44,t43,t44,J2zz+J3zz+t16+t17+t18+t19+t20+t21+t22+t23+t24+t32+t33+t40,t42,t43,t42,J3zz+t18+t20+t22+t24],[3,3]);
            obj.M_full = M_;
            obj.M_collocated = "";
            obj.M = casadi.Function('M', {obj.x(1:length(obj.x)/2), obj.v}, {M_});
        end
        
        function createC(obj)
            %    C_ = MYFILE1(L1,L2,L3,C2X,C3X,C2Y,C3Y,DQ1,DQ2,DQ3,M2,M3,Q2,Q3)
            q1   = obj.x(1);
            q2   = obj.x(2);
            q3   = obj.x(3);
            dq1  = obj.x(4);
            dq2  = obj.x(5);
            dq3  = obj.x(6);
            J1zz = obj.J1zz_;
            J2zz = obj.J2zz_;
            J3zz = obj.J3zz_;
            L1   = obj.L1_;
            L2   = obj.L2_;
            L3   = obj.L3_;
            c1x  = obj.c1x_;
            c2x  = obj.c2x_;
            c3x  = obj.c3x_;
            c1y  = obj.c1y_;
            c2y  = obj.c2y_;
            c3y  = obj.c3y_;
            m1   = obj.m1_;
            m2   = obj.m2_;
            m3   = obj.m3_;
            
            t2 = cos(q2);
            t3 = cos(q3);
            t4 = sin(q2);
            t5 = sin(q3);
            t6 = q2+q3;
            t7 = dq1+dq2+dq3;
            t8 = L3.*t5;
            t9 = c3y.*t3;
            t10 = cos(t6);
            t11 = c3x.*t5;
            t12 = sin(t6);
            t13 = c2y.*m2.*t2;
            t14 = c2x.*m2.*t4;
            t18 = L2.*m2.*t4;
            t19 = L2.*m3.*t4;
            t15 = L2.*t8;
            t16 = L2.*t9;
            t17 = L2.*t11;
            t20 = L1.*L3.*t12;
            t21 = L1.*c3y.*t10;
            t22 = L1.*c3x.*t12;
            t23 = L3.*m3.*t12;
            t24 = c3y.*m3.*t10;
            t25 = c3x.*m3.*t12;
            t26 = t8+t9+t11;
            t27 = t15+t16+t17+t20+t21+t22;
            t29 = t13+t14+t18+t19+t23+t24+t25;
            t28 = dq3.*m3.*t27;
            t30 = L1.*dq2.*t29;
            C_ = [-dq1.*(t28+t30)-t7.*t28-dq2.*(t28+t30+L1.*dq1.*t29);dq1.*(L1.*dq1.*t13+L1.*dq1.*t14+L1.*dq1.*t18+L1.*dq1.*t19-dq3.*m3.*t15-dq3.*m3.*t16-dq3.*m3.*t17+dq1.*m3.*t20+dq1.*m3.*t21+dq1.*m3.*t22)-L2.*dq2.*dq3.*m3.*t26-L2.*dq3.*m3.*t7.*t26;dq1.*m3.*(dq1.*t15+dq1.*t16+dq2.*t15+dq1.*t17+dq2.*t16+dq2.*t17+dq1.*t20+dq1.*t21+dq1.*t22)+L2.*dq2.*m3.*t26.*(dq1+dq2)];
            obj.C_full = C_;
            obj.C_collocated ="";
            obj.C = casadi.Function('C', {obj.x, obj.v}, {C_});
        end
        
        function createGravity(obj)
            %G_SYM = MYFILE2(L1,L2,L3,C1X,C2X,C3X,C1Y,C2Y,C3Y,G0,M1,M2,M3,Q1,Q2,Q3)
            q1   = obj.x(1);
            q2   = obj.x(2);
            q3   = obj.x(3);
            dq1  = obj.x(4);
            dq2  = obj.x(5);
            dq3  = obj.x(6);
            J1zz = obj.J1zz_;
            J2zz = obj.J2zz_;
            J3zz = obj.J3zz_;
            L1   = obj.L1_;
            L2   = obj.L2_;
            L3   = obj.L3_;
            c1x  = obj.c1x_;
            c2x  = obj.c2x_;
            c3x  = obj.c3x_;
            c1y  = obj.c1y_;
            c2y  = obj.c2y_;
            c3y  = obj.c3y_;
            m1   = obj.m1_;
            m2   = obj.m2_;
            m3   = obj.m3_;
            g0   = 9.8;
            
            t2 = cos(q1);
            t3 = q1+q2;
            t4 = L1.*t2;
            t5 = cos(t3);
            t6 = q3+t3;
            t7 = sin(t3);
            t8 = cos(t6);
            t9 = sin(t6);
            t10 = L2.*t5;
            t11 = c2x.*t5;
            t12 = c2y.*t7;
            t13 = L3.*t8;
            t14 = c3x.*t8;
            t15 = c3y.*t9;
            t16 = -t12;
            t17 = -t15;
            g_ = [g0.*m2.*(t4+t10+t11+t16)+g0.*m3.*(t4+t10+t13+t14+t17)+g0.*m1.*(t4+c1x.*t2-c1y.*sin(q1));g0.*m3.*(t10+t13+t14+t17)+g0.*m2.*(t10+t11+t16);g0.*m3.*(t13+t14+t17)];
            obj.gravity_full = g_;
            obj.gravity_collocated = "";
            obj.gravity = casadi.Function('gravity', {obj.x(1:length(obj.x)/2), obj.v}, {g_});
        end
        
        function xDot = testXDot(obj, x, v_nominal, tau)
            p  = x(1:length(x)/2);
            dp = x(length(x)/2+1:end);
            xDot = inv(obj.M(p, v_nominal)) * (-obj.C(x, v_nominal)*dp - obj.gravity(p, v_nominal) + tau);
        end
    end
    
    
    
end