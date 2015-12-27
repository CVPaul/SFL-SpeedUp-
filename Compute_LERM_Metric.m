% give two set of data compute the kernel of each (With out centering
% operation)
% input- the Set1 and the Data Set2
% output- the LED distance matrix of Set1 and Set2
function outMatrix = Compute_LERM_Metric(Set1,Set2,simFlag)
if nargin<3
    simFlag = false; % the Kernel is set to be asymmetric as default
end
if (nargin < 2)
    Set2 = Set1;
    simFlag = true;
end

l1 = size(Set1,3);
l2 = size(Set2,3);
outMatrix = zeros(l2,l1);

if (simFlag)
    for tmpC1 = 1:l1
        for tmpC2 = tmpC1:l2
            Set_1_temp=logm(Set1(:,:,tmpC1));
            Set_2_temp=logm(Set2(:,:,tmpC2));
            temp=Set_1_temp(:)-Set_2_temp(:);
            outMatrix(tmpC2,tmpC1) = temp(:)'*temp(:);
            if  (outMatrix(tmpC2,tmpC1) < 1e-10)
                outMatrix(tmpC2,tmpC1) = 0.0;
            end
            outMatrix(tmpC1,tmpC2) = outMatrix(tmpC2,tmpC1);% here we do not deal wtih tmpC1==tmpC2 it's OK
        end
    end
else
    for tmpC1 = 1:l1
        for tmpC2 = 1:l2
            Set_1_temp=logm(Set1(:,:,tmpC1));
            Set_2_temp=logm(Set2(:,:,tmpC2));
            temp=Set_1_temp(:)-Set_2_temp(:);
            outMatrix(tmpC2,tmpC1) = temp(:)'*temp(:);
            if  (outMatrix(tmpC2,tmpC1) < 1e-10)
                outMatrix(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end

end