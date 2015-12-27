% compute the gradient for each kernel term K_{ij},which means for each
% K_{ij}=tr(log(W'*X*W)log(W'*Y*W)),we need to compute this: d/d (K_ij)
% input- U: the param W matrix
%        covD_Struct: input data:X, lable:Y and term to use mask:G and
%        rank of the param matrix:r
% output- nCost: the Cost value
%         nGrad: the gradient of the loss function
function [nCost,nGrad]=compute_LERM_Sim_CostGrad(U,trnStruct,beta)
    nPoints = length(trnStruct.y);
    n=size(trnStruct.X(:,:,1),1);
    XU = zeros(n,trnStruct.r,nPoints);
    for tmpC1 = 1:nPoints
        XU(:,:,tmpC1)=trnStruct.X(:,:,tmpC1)*U;
    end
    OptStruct=SFL_OptSupport_OptStruct(trnStruct.X,U);
    hK=Compute_LERM_Sim(OptStruct.H,OptStruct.H,true,beta);
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
%     Z_W=(YK/hK_norm-nCost/hK_norm/hK_norm*hK);

    nGrad=0;
    for i=1:nPoints
        for j=i+1:nPoints
            if(trnStruct.G(i,j)==0) 
                continue;
            end 

            nGrad=nGrad+(trnStruct.G(i,j)+trnStruct.G(j,i))*Z_W(i,j)*(-beta)*(hK0(i,j))*...
                (4*XU(:,:,i)*...
                dlogm(OptStruct.U(:,:,i),OptStruct.invU(:,:,i),OptStruct.Z(:,:,i),OptStruct.H(:,:,i)-OptStruct.H(:,:,j))+...
                4*XU(:,:,j)*...
                dlogm(OptStruct.U(:,:,j),OptStruct.invU(:,:,j),OptStruct.Z(:,:,j),OptStruct.H(:,:,j)-OptStruct.H(:,:,i)));
        end
    end
    nCost=-nCost;
    nGrad=-nGrad;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D=dlogm(Ut,invUt,Z,H)
    Hj=invUt*H*Ut;
    D=Ut*(Hj.*Z)*invUt;
end
% function D = dlogm(X, H)
% % function D = dlogm(X, H)
% %
% % Code to compute the directional derivative (the Fr¨¦chet derivative) of
% % the matrix logarithm logm at X along H, according to a very nice trick
% % Nick Higham talked mentioned in a talk I attended. This code will also
% % work for expm, sqrtm, ... with appropriate modifications.
% %
% % Nicolas Boumal, UCLouvain, June 24, 2014.
% 
%     n = size(X, 1);
% 
%     assert(length(size(X)) == 2,     'X and H must be square matrices.');
%     assert(length(size(H)) == 2,     'X and H must be square matrices.');
%     assert(size(X, 1) == size(X, 2), 'X and H must be square matrices.');
%     assert(all(size(X) == size(H)),  'X and H must have the same size.');
% 
%     Z = zeros(n);
%     A = logm([X, H ; Z, X]);
%     D = A(1:n, (n+1):end);
% 
% end
%             