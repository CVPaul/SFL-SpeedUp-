% convert the data to appointed format
function covD_Struct=convert_data(filename,rank,with_pca)
load(filename);
%% PCA-----------------------------------------------------------
if(exist('with_pca','var')&&with_pca==true)
    PCARatio=0.95;
    [x_mean, x_var, W_PCA, eig_value, Gallery_pca]=PCA(Gallery.X,PCARatio);
    Probe_PCA=PCAProjction(x_mean,x_var,W_PCA,Probe.X);
    Gallery.X=Gallery_pca;
    Probe.X=Probe_PCA;
end
%%-------------------------------End--------------------------------------
covD_Struct.trn_X=Compute_Cov(Gallery.X,true);
covD_Struct.trn_y=Gallery.y;
covD_Struct.tst_X=Compute_Cov(Probe.X,true);
covD_Struct.tst_y=Probe.y;
covD_Struct.r=rank;
covD_Struct.n= size(covD_Struct.trn_X,1);