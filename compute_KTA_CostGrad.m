% compute the gradient for each kernel term K_{ij},which means for each
% K_{ij}=tr(log(W'*X*W)log(W'*Y*W)),we need to compute this: d/d (K_ij)
% input- U: the param W matrix
%        covD_Struct: input data:X, lable:Y and term to use mask:G and
%        rank of the param matrix:r
% output- nCost: the Cost value
%         nGrad: the gradient of the loss function
function [nCost,nGrad]=compute_KTA_CostGrad(U,trnStruct)
    nPoints = length(trnStruct.y);
    n=size(trnStruct.X(:,:,1),1);
    % I_r = eye(covD_Struct.r);
    Log_UXU = zeros(trnStruct.r,trnStruct.r,nPoints);
    % inv_UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
    UXU = zeros(trnStruct.r,trnStruct.r,nPoints);
    XU = zeros(n,trnStruct.r,nPoints);
    for tmpC1 = 1:nPoints
        XU(:,:,tmpC1)=trnStruct.X(:,:,tmpC1)*U;
        UXU(:,:,tmpC1) = U'*XU(:,:,tmpC1);
        Log_UXU(:,:,tmpC1) = logm(UXU(:,:,tmpC1));
    %     inv_UXU(:,:,tmpC1) = I_r/UXU(:,:,tmpC1);
    end
    hK=Compute_Sim_Matrix(Log_UXU);
    hK=trnStruct.G.*hK;
    H=eye(nPoints)-ones(nPoints)/nPoints;
    hK=H*hK*H; % centering
    
    % compute YY^{T}
    m=length(trnStruct.y);
    Y=full(sparse(1:m,trnStruct.y,1)); % convert the lables to matrixs format
%     YK=2*(Y*Y')-1; 
    YK=Y*Y';
    YK=trnStruct.G.*YK;
    YK=H*YK*H;
    
    YK_norm=sqrt(YK(:)'*YK(:)); % normalization
    YK=YK/YK_norm;

    hK_norm=sqrt(hK(:)'*hK(:));

    nCost=hK(:)'*YK(:)/hK_norm;
    
    Z_W=H*(YK/hK_norm-nCost/hK_norm/hK_norm*hK)*H;

    nGrad=0;
    for i=1:nPoints
        if(trnStruct.G(i,i)==1)
             nGrad=nGrad+Z_W(i,i)*trnStruct.G(i,i)*...
                (4*XU(:,:,i)*dlogm(UXU(:,:,i),Log_UXU(:,:,i)));
        end
        for j=i+1:nPoints
            if(trnStruct.G(i,j)==0) 
                continue;
            end 
            % Bi=2*XU(:,:,i); Bj=2*XU(:,:,j);
            nGrad=nGrad+(trnStruct.G(i,j)+trnStruct.G(j,i))*Z_W(i,j)*...
                (2*XU(:,:,i)*dlogm(UXU(:,:,i),Log_UXU(:,:,j))+...
                2*XU(:,:,j)*dlogm(UXU(:,:,j),Log_UXU(:,:,i)));
        end
    end
    nCost=-nCost;
    nGrad=-nGrad;
end

function D = dlogm(X, H)
% function D = dlogm(X, H)
%
% Code to compute the directional derivative (the Fr¨¦chet derivative) of
% the matrix logarithm logm at X along H, according to a very nice trick
% Nick Higham talked mentioned in a talk I attended. This code will also
% work for expm, sqrtm, ... with appropriate modifications.
%
% Nicolas Boumal, UCLouvain, June 24, 2014.

    n = size(X, 1);

    assert(length(size(X)) == 2,     'X and H must be square matrices.');
    assert(length(size(H)) == 2,     'X and H must be square matrices.');
    assert(size(X, 1) == size(X, 2), 'X and H must be square matrices.');
    assert(all(size(X) == size(H)),  'X and H must have the same size.');

    Z = zeros(n);
    A = logm([X, H ; Z, X]);
    D = A(1:n, (n+1):end);

end
            