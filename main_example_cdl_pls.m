clear;
clc;
% load('Toy_example.mat');
path = '..\A-data-YTC';
fst=dir([path,'\*.mat']);
n=length(fst);
fRate1 = zeros(n,1);
load('YTC_KTA_result_Metric_3_dim50_result.mat');
for iter = 1 : n
    %% Data Pre-Process
    ff = fst(iter).name;
    load([path '\' ff]);
    fprintf('PLS run on %s\n',ff);
    
    PCARatio = 0.95;
    [x_mean, x_var, W_pca,eig_value, Gallery_pca] = PCA(Gallery.X,PCARatio);
    Probe_pca = PCAProjction(x_mean, x_var, W_pca,Probe.X);    

    C = Compute_Cov(Gallery_pca,true);
    LogC=compute_Log_W_Cov(Param_W{iter},C);
%     LogC = Compute_Log_Cov(C);
    
%     K = Compute_Riemann_Kernel(LogC,[]);
    K=Compute_Sim_Matrix(LogC,LogC,true,true);
    
    C_t = Compute_Cov(Probe_pca,true);
    LogC_t=compute_Log_W_Cov(Param_W{iter},C_t);
%     LogC_t = Compute_Log_Cov(C_t);
    
%     K_t = Compute_Riemann_Kernel(LogC_t,LogC); 
    K_t=Compute_Sim_Matrix(LogC_t,LogC,false,true);
    K_t=K_t';
    %%
    %% KPLS - (centralized model) %%%%%%
    %%%% centralization K, K_t, (centralization of Y and Yt already done above)
    y = Gallery.y;
    y_t = Probe.y;
    n = length(y);
    nt = length(y_t);
%     M=eye(n)-ones(n,n)/n;
%     Mt=ones(nt,n)/n;
%     K_t = (K_t - Mt*K)*M;
%     K=M*K*M;
    
    Y_unique = unique(Gallery.y);
    Y = zeros(n,length(Y_unique));
    for i = 1 : n
        Y(i,y(i)) = 1;
    end
    
    %%%% number of used latent vectors (componets)
    Fac=length(Y_unique)-1; %Fac=10
    [B,T]=KerNIPALS(K,Y,Fac,0);    %%% a) NIPALS based KPLS
    % [B,T] = KerPLS_eig(K,Y,Fac,0); %%% b) K*Y*Y'*t = a *t based KPLS
    % [B,T,U]=KerSIMPLS1(K,Y,Fac);   %%% c) Kernel SIMPLS for single output (this equals a) and b))
    
    %% Nearest Neighbour Clasfication
    Yt_hat=K_t*B;
    corrNum = 0;
    for i = 1 : nt
        [sort_y ind_y] = sort(Yt_hat(i,:),'descend');
        if(ind_y(1)==y_t(i))
            corrNum = corrNum + 1;
        end        
    end
    fRate1(iter) = corrNum/nt;
    fprintf('fRate = %f \n', fRate1(iter));
end
[zf1 muf1 stdf1] = zscore(fRate1);
save('result','fRate1','muf1','stdf1');
clear
