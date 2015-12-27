% this part is used to compute the numerical gradient of the KTA function
% for fast computation,create on 2015-04-22
% compute the numerical gradient for each kernel term K_{ij},which means 
% for each K_{ij}=tr(log(W'*X*W)log(W'*Y*W)),we need to compute this:(f(G+h*H)-f(G-hH))/2
% input- U: the param W matrix
%        covD_Struct: input data:X, lable:Y and term to use mask:G and
%        rank of the param matrix:r
% output- nCost: the Cost value
%         nGrad: the gradient of the loss function
function [nCost,nGrad]=numerical_KTA_CostGrad(U,covD_Struct)
    delta=1e-4;
    [m,n] = size(U);
    nCost=compute_KTA_Cost(U,covD_Struct);
    nGrad=zeros(m,n);
    for i=1:m
        for j=1:n
            T0=U; T0(i,j) = T0(i,j) -delta;
            T1=U; T1(i,j)  = T1(i,j) +delta;
            f0=compute_KTA_Cost(T0,covD_Struct);
            f1=compute_KTA_Cost(T1,covD_Struct);
            nGrad(i,j) = (f1-f0) / (2*delta);
        end
    end
end
function cost=compute_KTA_Cost(U,covD_Struct)
    nPoints = length(covD_Struct.y);
    n=size(covD_Struct.X(:,:,1),1);
    Log_UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
    UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
    XU = zeros(n,covD_Struct.r,nPoints);
    for tmpC1 = 1:nPoints
        XU(:,:,tmpC1)=covD_Struct.X(:,:,tmpC1)*U;
        UXU(:,:,tmpC1) = U'*XU(:,:,tmpC1);
        Log_UXU(:,:,tmpC1) = logm(UXU(:,:,tmpC1));
    end
    hK=Compute_Kernel_Matrix(Log_UXU);
    H=eye(nPoints)-ones(nPoints)/nPoints;
    hK=H*hK*H; % centering

    % compute YY^{T}
    m=length(covD_Struct.y);
    Y=full(sparse(1:m,covD_Struct.y,1)); % convert the lables to matrixs format
    YK=H*Y;
    YK=YK*YK';
    YK_norm=sqrt(YK(:)'*YK(:)); % normalization
    YK=YK/YK_norm;

    hK_norm=sqrt(hK(:)'*hK(:));
    cost=hK(:)'*YK(:)/hK_norm;
end