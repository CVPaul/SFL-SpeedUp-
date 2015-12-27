% kernel learning with KTA Criterion on Riemannain manifold
% -----------------------new function added----2015-04-23------------------
clc;
close all;
clear all;
addpath(genpath('.'));
% load('toy_data');
path='..\A-data-YTC-pca';
fst=dir([path '/*.mat']);
n=length(fst);
fprintf('%d files in the folder(%s)\n',n,path);
for k=1:n
    fprintf('the %d-th file in the path is %s\n',k,fst(k).name);
end
Metric_Flag = 4; %1:AIRM, 2:Stein, 3:LED 4: LED_distSim
graph_kw = 3;
graph_kb = 3; 
newDim = 70;
maxiter=50;
%% main loop
global main_iter; 
fRate=zeros(n,2);
Param_W=cell(n,1);
prefix='YTC_SFL_result';
% Metric : LED, dim(M1) = 4656, dim(M2) = 210. 
% Recognition accuracy for the high-dimensional manifold (M1)-> 62.6%.
% Recognition accuracy after learning the low-dimensional manifold (M2)-> 55.9%.
for main_iter=1:n
    ff=fst(main_iter).name;
    fprintf('Current run on %s\n',ff);
    filename=[path '\' ff];% for windows
    covD_Struct=convert_data(filename,newDim,true); %% with pca
%     covD_Struct=Construct_HDM_UIUC_data(filename,newDim);
    U = eye(covD_Struct.n,covD_Struct.r); 
    [acc,W]=Manifold_Learning(U,covD_Struct,maxiter,Metric_Flag ,newDim,graph_kw,graph_kb);
    fRate(main_iter,:)=acc;
    Param_W{main_iter}=W;
end
ML_mean=mean(fRate);
ML_std=std(fRate);
fprintf('final result: mean_acc=%f, std=%f\n',ML_mean,ML_std);
filename=[prefix '_Metric_' num2str(Metric_Flag) '_dim' num2str(newDim) '_result_[v_b=20].mat'];
save(filename,'fRate','ML_mean','ML_std','Param_W');