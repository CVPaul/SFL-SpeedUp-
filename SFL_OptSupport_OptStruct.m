% create on 2015-12-18:Approximate method 2 for PLS on Riemannian manifold
% preprocess function, this function will prepare all the data for the next
% optimal step:
% input: trn_X- the training spd matrix
%        A0- the support point, estimated at the last time
%        eps- the tolerance to judge if two float numbers are equal, the
%        default value of eps is 1e-8.
% output:OptStruct- contains data for the next optimal step and include the
% following things:
%           OptStruct.B- B=A*trnX,define S as the eigenvalue matrix of B
%           OptStruct.U- B=U*S*U^{-1},the eigenvector matrix of B
%           OptStruct.invU- invert of matrix U
%           OptStruct.H- H=logm(B)=U*logm(S)*U^{-1}
%           OptStruct.Z- the is construc from S, detail can be find in the paper
function OptStruct=SFL_OptSupport_OptStruct(trn_X,A0,eps)
    if nargin<3
        eps=1e-8;
    end
    n=size(trn_X,3);
    m=size(A0,2);
    OptStruct.B=zeros(m,m,n);
    OptStruct.U=zeros(m,m,n);
    OptStruct.invU=zeros(m,m,n);
    OptStruct.H=zeros(m,m,n);
    OptStruct.Z=zeros(m,m,n);
    for k=1:n
        OptStruct.B(:,:,k)=A0'*trn_X(:,:,k)*A0;
        [Ut,St]=eig(OptStruct.B(:,:,k));
        OptStruct.U(:,:,k)=Ut;
        OptStruct.invU(:,:,k)=eye(m)/Ut;
        st=diag(St);
        lst=log(st);
        OptStruct.H(:,:,k)=OptStruct.U(:,:,k)*diag(lst)*OptStruct.invU(:,:,k);
        Zt=diag(1./st);
        for i=1:m
            for j=i+1:m
                if abs(st(i)-st(j))<eps
                    Zt(i,j)=1/st(i);
                else
                    Zt(i,j)=(lst(i)-lst(j))/(st(i)-st(j));
                end
                Zt(j,i)=Zt(i,j);
            end
        end
        OptStruct.Z(:,:,k)=Zt;
    end
end