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
    fprintf('LDA run on %s\n',ff);
       
    PCARatio = 0.95;
    [x_mean, x_var, W_pca,eig_value, Gallery_pca] = PCA(Gallery.X,PCARatio);
    Probe_pca = PCAProjction(x_mean, x_var, W_pca,Probe.X);    
    
    Gallery.X=Gallery_pca;
    Probe.X=Probe_pca;
    
    C = Compute_Cov(Gallery_pca,true);
    LogC=compute_Log_W_Cov(Param_W{iter},C);
    K_V = Compute_Sim_Matrix(LogC,LogC,true,false);
    
    
    options = [];
    options. Kernel = 1;
    gnd = Gallery.y;
    [eigvector_kda, eigvalue_kda] = KDA(options,gnd,K_V);
    
    %%
    
    C_t = Compute_Cov(Probe.X,true);
    LogC_t=compute_Log_W_Cov(Param_W{iter},C_t);
%     LogC_t = Compute_Log_Cov(C_t);
    
    K_V_t = Compute_Sim_Matrix(LogC_t,LogC,false,false);
    K_V_t=K_V_t';
    
    Gallery_cdl = K_V*eigvector_kda;
    Probe_cdl = K_V_t*eigvector_kda;
    sim_mat = 1- pdist2(Gallery_cdl,Probe_cdl,'cosine');
    
    
    
    sampleNum = length(Probe.y);
    [sim ind] = sort(sim_mat,1,'descend');
    correctNum = length(find((Probe.y-Gallery.y(ind(1,:)))==0));
    fRate1(iter) = correctNum/sampleNum;
    fprintf('fRate = %f \n', fRate1(iter));
end
[zf1 muf1 stdf1] = zscore(fRate1);
save('result','fRate1','muf1','stdf1');
clear
