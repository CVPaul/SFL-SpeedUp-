% this function is design for HDM and UIUC,which has the original data save as
% covariance matrix,construct these data to covD_Struct struct for
% experiment use
% construct the data to appointed format
function covD_Struct=Construct_HDM_UIUC_data(filename,rank)
load(filename);
covD_Struct.trn_X=Process_Covs(Gallery.X);
covD_Struct.trn_y=Gallery.y;
covD_Struct.tst_X=Process_Covs(Probe.X);
covD_Struct.tst_y=Probe.y;
covD_Struct.r=rank;
covD_Struct.n= size(covD_Struct.trn_X,1);