% compute logm(W'*Cov*W)
% input- matrix W and covaraince samples
% output- log(W*Cov*W)
function log_WCovW=compute_Log_W_Cov(W,Covs)
samples=size(Covs,3);
r=size(W,2);
log_WCovW=zeros(r,r,samples);
for k=1:samples
    log_WCovW(:,:,k)=logm(W'*Covs(:,:,k)*W);
end