% compute the gradient for each kernel term K_{ij},which means for each
% K_{ij}=tr(log(W'*X*W)log(W'*Y*W)),we need to compute this: d/d (K_ij)
% input- U: the param W matrix
%        covD_Struct: input data:X, lable:Y and term to use mask:G and
%        rank of the param matrix:r
% output- nCost: the Cost value
%         nGrad: the gradient of the loss function
function [nCost,nGrad]=compute_Stein_Sim_CostGrad(U,trnStruct,beta)
    nPoints = length(trnStruct.y);
    n=size(trnStruct.X(:,:,1),1);
    I_r = eye(trnStruct.r);
    inv_UXU = zeros(trnStruct.r,trnStruct.r,nPoints);
    UXU = zeros(trnStruct.r,trnStruct.r,nPoints);
    XU = zeros(n,trnStruct.r,nPoints);
    for tmpC1 = 1:nPoints
        XU(:,:,tmpC1)=trnStruct.X(:,:,tmpC1)*U;
        UXU(:,:,tmpC1) = U'*XU(:,:,tmpC1);
        inv_UXU(:,:,tmpC1) = I_r/UXU(:,:,tmpC1);
    end
    hK=Compute_Stein_Sim(UXU,UXU,true,beta);
    hK0 = hK;

    hK=trnStruct.G.*hK;
    H=eye(nPoints)-ones(nPoints)/nPoints;
    hK=H*hK*H; % centering
    
    % compute YY^{T}
    m=length(trnStruct.y);
    Y=full(sparse(1:m,trnStruct.y,1)); % convert the lables to matrixs format
    YK=2*(Y*Y')-1; 
%     YK=Y*Y';
    YK=trnStruct.G.*YK;
    YK=H*YK*H;
    
    YK_norm=sqrt(YK(:)'*YK(:)); % normalization
    YK=YK/YK_norm;

    hK_norm=sqrt(hK(:)'*hK(:));

    nCost=hK(:)'*YK(:)/hK_norm;
    
    Z_W=H*(YK/hK_norm-nCost/hK_norm/hK_norm*hK)*H;
%         Z_W=(YK/hK_norm-nCost/hK_norm/hK_norm*hK);


    nGrad=0;
    for i=1:nPoints
        X_i = trnStruct.X(:,:,i);
        
        for j=i+1:nPoints
            if(trnStruct.G(i,j)==0)
                continue;
            end
            X_j = trnStruct.X(:,:,j);
            
            X_ij = 0.5*(X_i + X_j);
            
            
            nGrad=nGrad+(trnStruct.G(i,j)+trnStruct.G(j,i))*Z_W(i,j)*...
                (-beta)*(hK0(i,j))*(2*(X_ij*U)/(U'*X_ij*U)  ...
                    - (X_i*U)*inv_UXU(:,:,i) - (X_j*U)*inv_UXU(:,:,j));

        end
    end
    nGrad=-nGrad;
    nCost=-nCost;
end

