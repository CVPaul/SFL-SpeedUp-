% this function is design for HDM and UIUC,which has the original data save as
% covariance matrix
function CY1 = Process_Covs(SY1)
number_sets1=length(SY1);
n=size(SY1{1},1);
CY1=zeros(n,n,number_sets1);
for tmpC1=1:number_sets1 
    
    Y1=SY1{tmpC1};
%     y1_mu = mean(Y1,2);        
%     
%     Y1 = Y1-repmat(y1_mu,1,size(Y1,2));
%     Y1 = Y1*Y1'/(size(Y1,2)-1);
%     lamda = 0.001*trace(Y1);
%     Y1 = Y1+lamda*eye(size(Y1,1));
   
    
%     Y1 = det(Y1)^(-1/(size(Y1,1)+1))*[Y1+y1_mu*y1_mu' y1_mu;y1_mu' 1];
% %     lamda = 0.001*trace(Y1);
%     Y1 = Y1+lamda*eye(size(Y1,1));   
    CY1(:,:,tmpC1)= Y1+0.001*trace(Y1)*eye(size(Y1,1));
end
end