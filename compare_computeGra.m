% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

function [outCost,outGrad,covD_Struct,beta] = compute_AIRM_Sim_CostGrad(U,covD_Struct,beta)
%     outCost = 0;
    dF = zeros(size(U));
    nPoints = length(covD_Struct.y);
    I_r = eye(covD_Struct.r);
    UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
    inv_UXU = zeros(covD_Struct.r,covD_Struct.r,nPoints);
    for tmpC1 = 1:nPoints
        UXU(:,:,tmpC1) = U'*covD_Struct.X(:,:,tmpC1)*U;
        inv_UXU(:,:,tmpC1) = I_r/UXU(:,:,tmpC1);
    end
    %     hK=Compute_Sim_Matrix(Log_UXU);
    [hK, beta]=Compute_AIRM_Sim(UXU,UXU,true,beta);
    simK=hK;
    hK=covD_Struct.G.*hK;
%     H=eye(nPoints)-ones(nPoints)/nPoints;
%     hK=H*hK*H; % centering
    
    % compute YY^{T}
    m=length(covD_Struct.y);
    Y=full(sparse(1:m,covD_Struct.y,1)); % convert the lables to matrixs format
    YK=2*(Y*Y')-1; 
%     YK=Y*Y';
    YK=covD_Struct.G.*YK;
%     YK=H*YK*H;
    
    YK_norm=sqrt(YK(:)'*YK(:)); % normalization
    YK=YK/YK_norm;

    hK_norm=sqrt(hK(:)'*hK(:));

    outCost=hK(:)'*YK(:)/hK_norm;
    
%     Z_W=H*(YK/hK_norm-nCost/hK_norm/hK_norm*hK)*H;
    Z_W=(YK/hK_norm-outCost/hK_norm/hK_norm*hK);
    for i = 1:nPoints
        X_i = covD_Struct.X(:,:,i);
        for j = 1:nPoints
            if (covD_Struct.G(i,j) == 0)
                continue;
            end
            X_j = covD_Struct.X(:,:,j);
            %AIRM
    %         outCost = outCost + covD_Struct.G(i,j)*Compute_AIRM_Metric(UXU(:,:,i) , UXU(:,:,j));
            log_XY_INV = logm(UXU(:,:,i)*inv_UXU(:,:,j));

            dF = dF + 4*(-beta*simK(i,j))*covD_Struct.G(i,j)*Z_W(i,j)*((X_i*U)*inv_UXU(:,:,i)  ...
                -(X_j*U)*inv_UXU(:,:,j) )*log_XY_INV;
        end
    end
% outGrad = (eye(size(U,1)) - U*U')*dF;
    outGrad=-dF;
    outCost=-outCost;
end