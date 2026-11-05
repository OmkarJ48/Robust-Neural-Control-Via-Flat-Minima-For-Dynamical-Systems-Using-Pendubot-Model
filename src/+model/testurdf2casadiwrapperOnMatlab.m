% test forward dynamics idintree vs urdf2casadiWrapper integrated
clear 
clc 
close all

addpath('C:\Users\valer\matlab_toolbox\casadi-windows-matlabR2016a-v3.5.5');
import casadi.*

%% Choose a urdf model idyntree
location_tests_folder = pwd;
twoLink_urdf     = [location_tests_folder,'/urdf/twoLinks.urdf'];
twoLink_env      = [location_tests_folder,'/urdf/twoLinks_env.urdf'];
kuka_kr210       = [location_tests_folder,'/urdf/kuka_kr210.urdf'];
kuka_kr210_env   = [location_tests_folder,'/urdf/kuka_kr210_env.urdf'];

%% Input urdf file to acquire robot structure for idyn
robotURDFModelIDyn=kuka_kr210;

%% Constants
gravityModulus = 9.80665;
tol = 1e-10;

%% wrapperurdf2casadi
[v,x,u,xdot,v_nominal,~,~,~,~,urdf_model]=urdf2casadiWrapperModelConf();  % src/examples/urdf2casadiWrapperModelConf.m

nrOfJoints = urdf_model.smds_.NB;
g = [0;0;-gravityModulus];


%% Prepare variables to store results
nrOfTests = 100;

e_accID   = zeros(nrOfJoints,nrOfTests);
for i= 1:nrOfTests
    %% Random points
    jointPos = rand(nrOfJoints,1);
    jointVel = rand(nrOfJoints,1);
    jointAcc = rand(nrOfJoints,1);
    tau      = rand(nrOfJoints,1);
    
    %% IDynTree values 
    % Compute mass matrix with IDynTree
    M_IDyn = computeMassMatrixIDynTree(robotURDFModelIDyn,jointPos,jointVel,jointAcc,gravityModulus);
    % Compute gravity torques IDynTree
    %tau_gravity_IDyn = computeGravityTorqueIDynTree(robotURDFModel,jointPos,gravityModulus);
    % Compute the generalized bias force C(q,qd)*qd + g(q)
    genealizedBias_IDyn = computeGeneralizedBiasForceIDynTree(robotURDFModelIDyn,jointPos,jointVel,gravityModulus);
    
    %% compute direct dynamics IDyn
    acc_IDyn = M_IDyn\(-genealizedBias_IDyn  + tau);
%     
    %% Compute direct dynamics symbolic
    if(isempty(v))
        state = [jointPos;jointVel];
        xdot_func = casadi.Function('xdot_func', {x, u}, {urdf_model.qdd_});
        acc_ID = full(xdot_func(state,tau));
    else
        state = [jointPos;jointVel];
        xdot_func = casadi.Function('xdot_func', {v, x, u}, {urdf_model.qdd_});
        acc_ID = full(xdot_func(v_nominal,state,tau));
    end

    %% Compare iDynTee and symbolic 
    e_accID(:,i) = abs(acc_ID-acc_IDyn);
    
end
%% plot results
plot(e_accID);title('forward dynamics error norm: norm(acc_{symb} - acc_{IDyn})');legend
