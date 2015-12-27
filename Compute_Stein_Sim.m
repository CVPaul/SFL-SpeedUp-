% Author:
% - Mehrtash Harandi (mehrtash.harandi at gmail dot com)
%
% This file is provided without any warranty of
% fitness for any purpose. You can redistribute
% this file and/or modify it under the terms of
% the GNU General Public License (GPL) as published
% by the Free Software Foundation, either version 3
% of the License or (at your option) any later version.

function [outStein,beta] = Compute_Stein_Sim(Set1,Set2,simFlag,beta)

if nargin<3
    simFlag = false; % the Kernel is set to be asymmetric as default
end
if (nargin < 2)
    Set2 = Set1;
    simFlag = true;
end

l1 = size(Set1,3);
l2 = size(Set2,3);
outStein = zeros(l2,l1);

if (simFlag)
    for tmpC1 = 1:l1
        X = Set1(:,:,tmpC1);
        for tmpC2 = tmpC1+1:l2
            Y = Set2(:,:,tmpC2);
            outStein(tmpC2,tmpC1) = sqrt(log(det(0.5*(X+Y))) -  0.5*(log(det(X*Y))));
            if  (outStein(tmpC2,tmpC1) < 1e-10)
                outStein(tmpC2,tmpC1) = 0.0;
            end
            outStein(tmpC1,tmpC2) = outStein(tmpC2,tmpC1);
        end
    end
    
else
    for tmpC1 = 1:l1
        X = Set1(:,:,tmpC1);
        for tmpC2 = 1:l2
            Y = Set2(:,:,tmpC2);
            outStein(tmpC2,tmpC1) = sqrt(log(det(0.5*(X+Y))) -  0.5*(log(det(X*Y))));
            if  (outStein(tmpC2,tmpC1) < 1e-10)
                outStein(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end
if nargin<4
    beta = 1/(2*mean(outStein(:))^2);
end
outStein=exp(-beta*outStein.^2);

end
