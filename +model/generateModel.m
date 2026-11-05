%% if you generate Jacob0 with rpy active is gonna take a big take to compute
close all
clear variables
clc

%% Instantiate a |CodeGenerator| class object

% change the function to change the robot for generating mex matrix
% rob = MdlLBR4pSimple();
% genkinonly = false;
% cGen = CodeGenerator(rob,'mex','genmfun','genmex');
% if(genkinonly)
%    %with this function i build only the kinematic symbolic rapresentation
%    cGen.genkinematic('norpy','noeul',false);
% else
%    %with this function i build all the symbolic rapresentation of the robot
%    cGen.geneverything('norpy','noeul',false);
% end
% % add the generated class to matlab path
% addpath(cGen.basepath);
% 
% % create an empty file to advise that a symbolic version of the robot exist
% rob_name = rob.name;
% rob_name(rob_name==' ')=[]; 
% fid = fopen( strcat(cGen.basepath,'/',rob_name,'_done.m'), 'wt' );


model.mdlPuma;
% I get the parameters from mdlPuma
env_param = [v_1];
tau   = sym('tau',[1,p560.n],'real');
sym_bot = p560.sym;
[q, qd] = sym_bot.gencoords();


direct_dynamics = sym_bot.accel(q,qd,tau);

matlabFunction(direct_dynamics,'File','dD');