% give two set of data compute the kernel of each (With out centering
% operation)
% input- the Set1 and the Data Set2
% output- the Kernel Matrix of Set1 and Set2
function SimMat = Compute_Sim_Matrix(Set1,Set2,simFlag,norm_and_mean)
if nargin<4
    norm_and_mean=false;
end
if nargin<3
    simFlag = true; % the Kernel is set to be symmetric as default
end
if (nargin < 2)
    Set2 = Set1;
end
l1 = size(Set1,3);
l2 = size(Set2,3);
SimMat = zeros(l2,l1);
if norm_and_mean
    set_mean=0;
    for tmpk = 1:l2
        set_mean=set_mean+Set2(:,:,tmpk);
    end
    set_mean=set_mean/l2;
end
if (simFlag)        
    for tmpC1 = 1:l1
        for tmpC2 = tmpC1:l2
            Set_1_temp=Set1(:,:,tmpC1);
            Set_2_temp=Set2(:,:,tmpC2);
            
            if norm_and_mean
                Set_1_temp=Set_1_temp-set_mean;
                Set_1_temp=Set_1_temp/sqrt(Set_1_temp(:)'*Set_1_temp(:));

                Set_2_temp=Set_2_temp-set_mean;
                Set_2_temp=Set_2_temp/sqrt(Set_2_temp(:)'*Set_2_temp(:));
            end
            
            SimMat(tmpC2,tmpC1) = Set_1_temp(:)'*Set_2_temp(:);
            if  (SimMat(tmpC2,tmpC1) < 1e-10)
                SimMat(tmpC2,tmpC1) = 0.0;
            end
            SimMat(tmpC1,tmpC2) = SimMat(tmpC2,tmpC1);% here we do not deal wtih tmpC1==tmpC2 it's OK
        end
    end
else
    for tmpC1 = 1:l1
        for tmpC2 = 1:l2
            Set_1_temp=Set1(:,:,tmpC1);
            Set_2_temp=Set2(:,:,tmpC2);
            
            if norm_and_mean
                Set_1_temp=Set_1_temp-set_mean;
                Set_1_temp=Set_1_temp/sqrt(Set_1_temp(:)'*Set_1_temp(:));

                Set_2_temp=Set_2_temp-set_mean;
                Set_2_temp=Set_2_temp/sqrt(Set_2_temp(:)'*Set_2_temp(:));
            end
            
            SimMat(tmpC2,tmpC1) = Set_1_temp(:)'*Set_2_temp(:);
            if  (SimMat(tmpC2,tmpC1) < 1e-10)
                SimMat(tmpC2,tmpC1) = 0.0;
            end
        end
    end
end
end