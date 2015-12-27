function CY1 = Compute_Cov(SY1,withmean)
samples=length(SY1);
m=size(SY1{1},1);
CY1=zeros(m+1,m+1,samples);
if(~withmean||nargin<2)
    for tmpC1=1:samples
        Y1=SY1{tmpC1};
        y1_mu = mean(Y1,2);        
        Y1 = Y1-repmat(y1_mu,1,size(Y1,2));
        Y1 = Y1*Y1'/(size(Y1,2)-1);
        lamda = 0.001*trace(Y1);
        Y1 = Y1+lamda*eye(size(Y1,1));
        CY1(:,:,tmpC1)= Y1;
    end
else %---------------------------------------------------------------------
    %% with informatio of mean
    fprintf('compute Covaraince With information of Mean\n')
    for index=1:samples
        X=SY1{index};
        n=size(X,2);
        meX=mean(X,2);
        X=X-repmat(meX,1,n);
        X=X*X'/(n-1);
        lamda=0.001*trace(X);
        X=X+lamda*eye(m);
        CY1(:,:,index)=det(X)^(-1/(m+1))*[X+meX*meX' meX;meX' 1];
    end
end
end