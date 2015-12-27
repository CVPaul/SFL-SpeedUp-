% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.
function [acc,U]=Manifold_Learning(U,covD_Struct,maxiter,Metric_Flag,newDim,graph_kw,graph_kb)
fprintf('print the peremeters for check!maxiter=%d,Dim=%d,v_w=%d,v_b=%d\n',maxiter,newDim,graph_kw,graph_kb);
% load([path '/' ff]);% for linux
%initializing training structure
trnStruct.X = covD_Struct.trn_X;
trnStruct.y = covD_Struct.trn_y;
trnStruct.n = covD_Struct.n;
trnStruct.nClasses = max(covD_Struct.trn_y);
trnStruct.r = newDim;
trnStruct.Metric_Flag = Metric_Flag;

%Generating graph
nPoints = length(trnStruct.y);
nPoints_te = size(covD_Struct.tst_X,3);

trnStruct.G = generate_Graphs(trnStruct.X,trnStruct.y,graph_kw,graph_kb,Metric_Flag);



%- different ways of initializing, the first 10 features are genuine so
%- the first initialization is the lucky guess, the second one is a random
%- attempt and the last one is the worst possible initialization.

% U = orth(rand(trnStruct.n,trnStruct.r));
% U = eye(trnStruct.n,trnStruct.r); 
% U = [zeros(trnStruct.n-trnStruct.r,trnStruct.r);eye(trnStruct.r)];


nPoints = length(trnStruct.y);
UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
for tmpC1 = 1:nPoints
    UXU(:,:,tmpC1) = U'*trnStruct.X(:,:,tmpC1)*U;
end

% Create the problem structure.
if(Metric_Flag==1)
    fprintf('AID similarity is used\n');
    [pair_dist, beta] = Compute_AIRM_Sim(UXU,UXU,true);
elseif(Metric_Flag==2)
    fprintf('Stein divergence is used\n');
    [pair_dist, beta] = Compute_Stein_Sim(UXU,UXU,true);
    beta = beta*1; %% attention!

elseif(Metric_Flag==4)
    fprintf('LED similarity is used\n');    
    [pair_dist, beta] =  Compute_LERM_Sim(UXU,UXU,true);
    beta = beta*2.5; %% attention!
else
    fprintf('metric is wrong!\n');  
end

%manifold = grassmannfactory(covD_Struct.n,covD_Struct.r);
manifold = symfixedrankYYfactory(covD_Struct.n,covD_Struct.r);
problem.M = manifold;


% conjugate gradient on Grassmann
problem.costgrad = @(U) supervised_WB_CostGrad(U,trnStruct,beta);
U  = conjugategradient(problem,U,struct('maxiter',maxiter));


TL_trnX = zeros(newDim,newDim,length(covD_Struct.trn_y));
for tmpC1 = 1:nPoints
    TL_trnX(:,:,tmpC1) = U'*covD_Struct.trn_X(:,:,tmpC1)*U;
end
TL_tstX = zeros(newDim,newDim,length(covD_Struct.tst_y));
for tmpC1 = 1:length(covD_Struct.tst_y)
    TL_tstX(:,:,tmpC1) = U'*covD_Struct.tst_X(:,:,tmpC1)*U;
end

if (Metric_Flag == 1)
    %AIRM
%     pair_sim = Compute_AIRM_Metric(covD_Struct.tst_X,covD_Struct.trn_X,false);
%     pair_sim_U = Compute_AIRM_Metric(TL_tstX,TL_trnX,false);

    [pair_dist_sim_tr, beta]=Compute_AIRM_Sim(covD_Struct.trn_X);
    pair_dist_sim=Compute_AIRM_Sim(covD_Struct.trn_X,covD_Struct.tst_X,false,beta);
    
    [pair_dist_sim_tr_U, beta]=Compute_AIRM_Sim(TL_trnX);
    pair_dist_sim_U=Compute_AIRM_Sim(TL_trnX,TL_tstX,false,beta);
elseif (Metric_Flag == 2)
    %Stein
%     pair_sim = Compute_Stein_Metric(covD_Struct.tst_X,covD_Struct.trn_X,false);
%     pair_sim_U = Compute_Stein_Metric(TL_tstX,TL_trnX,false);
    
    [pair_dist_sim_tr, beta]=Compute_Stein_Sim(covD_Struct.trn_X);
    beta = beta*1; %% attention!
    pair_dist_sim=Compute_Stein_Sim(covD_Struct.trn_X,covD_Struct.tst_X,false,beta);
    
    [pair_dist_sim_tr_U, beta]=Compute_Stein_Sim(TL_trnX);
    beta = beta*1; %% attention!
    pair_dist_sim_U=Compute_Stein_Sim(TL_trnX,TL_tstX,false,beta);
    
    
elseif(Metric_Flag==4)
    % LED
    log_trn_org=zeros(covD_Struct.n,covD_Struct.n,nPoints);
    log_tst_org=zeros(covD_Struct.n,covD_Struct.n,size(covD_Struct.tst_X,3));
    log_trn=zeros(covD_Struct.r,covD_Struct.r,nPoints);
    log_tst=zeros(covD_Struct.r,covD_Struct.r,size(covD_Struct.tst_X,3));
    for tk=1:nPoints   
        log_trn(:,:,tk)=logm(TL_trnX(:,:,tk));
        log_trn_org(:,:,tk)=logm(covD_Struct.trn_X(:,:,tk));
    end
    for tk=1:size(covD_Struct.tst_X,3)
        log_tst(:,:,tk)=logm(TL_tstX(:,:,tk));
        log_tst_org(:,:,tk)=logm(covD_Struct.tst_X(:,:,tk));
    end
%     pair_sim=Compute_Sim_Matrix(log_tst_org, log_trn_org,false,true);
%     pair_sim_U = Compute_Sim_Matrix(log_tst,log_trn,false,true);% set simFlag as false for test

    [pair_dist_sim_tr, beta]=Compute_LERM_Sim(log_trn_org);
    beta = beta*2.5; %% attention!

    pair_dist_sim=Compute_LERM_Sim(log_trn_org,log_tst_org,false,beta);

    [pair_dist_sim_tr_U, beta]=Compute_LERM_Sim(log_trn);
    beta = beta*2.5; %% attention!

    pair_dist_sim_U=Compute_LERM_Sim(log_trn,log_tst,false,beta);

else
    error('the metric is not defined');
end

N=eye(nPoints)-ones(nPoints,nPoints)/nPoints;
Nt=ones(nPoints_te,nPoints)/nPoints;
pair_dist_sim = (pair_dist_sim - Nt*pair_dist_sim_tr)*N;
pair_dist_sim_U = (pair_dist_sim_U - Nt*pair_dist_sim_tr_U)*N;
pair_sim = pair_dist_sim';
pair_sim_U = pair_dist_sim_U';


[~,maxIDX] = max(pair_sim);
y_hat = covD_Struct.trn_y(maxIDX);
CRR(1) = sum(covD_Struct.tst_y == y_hat)/length(covD_Struct.tst_y);

[~,maxIDX] = max(pair_sim_U);
y_hat = covD_Struct.trn_y(maxIDX);
CRR(2) = sum(covD_Struct.tst_y == y_hat)/length(covD_Struct.tst_y);


global main_iter;
fprintf('result on fold(%d):\n',main_iter);
if (Metric_Flag == 1)
    %AIRM
    fprintf('\n-----------------------------------------\n')
    fprintf('Metric : AIRM, dim(M1) = %d, dim(M2) = %d. \n',0.5*covD_Struct.n*(covD_Struct.n+1),0.5*newDim*(newDim+1));
    fprintf('Recognition accuracy for the high-dimensional manifold (M1)-> %.1f%%.\n',100*CRR(1));
    fprintf('Recognition accuracy after learning the low-dimensional manifold (M2)-> %.1f%%.\n',100*CRR(2));
    fprintf('-----------------------------------------\n')
elseif(Metric_Flag == 2)
    %Stein
    fprintf('\n-----------------------------------------\n')
    fprintf('Metric : Stein, dim(M1) = %d, dim(M2) = %d. \n',0.5*covD_Struct.n*(covD_Struct.n+1),0.5*newDim*(newDim+1));
    fprintf('Recognition accuracy for the high-dimensional manifold (M1)-> %.1f%%.\n',100*CRR(1));
    fprintf('Recognition accuracy after learning the low-dimensional manifold (M2)-> %.1f%%.\n',100*CRR(2));
    fprintf('-----------------------------------------\n')
else
    fprintf('\n-----------------------------------------\n')
    fprintf('Metric : LED, dim(M1) = %d, dim(M2) = %d. \n',0.5*covD_Struct.n*(covD_Struct.n+1),0.5*newDim*(newDim+1));
    fprintf('Recognition accuracy for the high-dimensional manifold (M1)-> %.1f%%.\n',100*CRR(1));
    fprintf('Recognition accuracy after learning the low-dimensional manifold (M2)-> %.1f%%.\n',100*CRR(2));
%     fprintf('Accuracy On trianning data after learning the low-dimensional manifold (M2)-> %.1f%%.\n',100*CRR(3));
%     fprintf('Accuracy On original trianning data high-dimensional manifold (M1)-> %.1f%%.\n',100*CRR(4));
    fprintf('-----------------------------------------\n')
end
acc=CRR;