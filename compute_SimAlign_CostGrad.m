% compute the gradient for each similarity term sim_{ij},which means for 
% each sim_{ij}=tr(log(W'*X*W)log(W'*Y*W))/(norm(log(W'*X*W)*norm(log(W'*Y*W))
% we need to compute this: d/d (sim_ij)
% input- U: the param W matrix
%        covD_Struct: input data:X, lable:Y and term to use mask:G and
%        rank of the param matrix:r
% output- nCost: the Cost value
%         nGrad: the gradient of the loss function
function [nCost,nGrad]=compute_SimAlign_CostGrad(U,covD_Struct)
    nPoints = length(covD_Struct.y);
    n=size(covD_Struct.X(:,:,1),1);
    UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
    XU = zeros(n,covD_Struct.r,nPoints);
    for tmpC1 = 1:nPoints
        XU(:,:,tmpC1)=covD_Struct.X(:,:,tmpC1)*U;
        UXU(:,:,tmpC1) = U'*XU(:,:,tmpC1);
    end
    sim=Compute_sim_Matrix(UXU);
    Y=full(sparse(1:m,covD_Struct.y,1)); % convert the lables to matrixs format
    YK=Y*Y';
    nCost=Y(sim
    
    covD_Struct.G=YK;
    nGrad=0;
    for i=1:nPoints % when i=j we have i
        for j=i+1:nPoints
            if(covD_Struct.G(i,j)==0) 
                continue;
            end 
            % Bi=2*XU(:,:,i); Bj=2*XU(:,:,j);
            nGrad=nGrad+(covD_Struct.G(i,j)+covD_Struct.G(j,i))*...
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
            