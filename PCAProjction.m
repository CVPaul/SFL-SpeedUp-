function  SY2 = PCAProjction(x_mean, x_var, W_pca,SY1)
%% %%%%%%%%%%%% Data Pre-Process %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nsets=size(SY1,3);
% nimgPerSet=size(SY1,2);
nsets=length(SY1);
X_pca = [];
for i = 1 : nsets
%     X_pca(:,(i-1)*nimgPerSet+1:i*nimgPerSet) = SY1(:,:,i);
    Y1 = SY1{i};
    X_pca = [X_pca Y1];
end
X_pca = ZeroMeanOneVar(X_pca, x_mean,x_var);
Y_PCA = W_pca' * X_pca;

% for i = 1 : nsets
%     SY2(:,:,i) = Y_PCA(:,(i-1)*nimgPerSet+1:i*nimgPerSet);   
% end

num_beg = 1;
SY2 = cell(nsets,1);
for i = 1 : nsets
%     SY2(:,:,i) = Y_PCA(:,(i-1)*nimgPerSet+1:i*nimgPerSet);   
    Y1 = SY1{i};
    n = size(Y1,2);
    SY2{i} = Y_PCA(:,num_beg:num_beg+n-1);    
    num_beg = num_beg + n;
end