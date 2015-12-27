function  [x_mean, x_var, W_pca,eig_value, SY2] = PCA(SY1,PCARatio)
%% Part 01: Data Pre-Process
nsets=length(SY1);
X_pca = [];
for i = 1 : nsets
    Y1 = SY1{i};
    X_pca = [X_pca Y1]; % since we dont konw the correct number of the each set so ...
end

[dim, sample_num]= size(X_pca);
[x_mean, x_var] = GetZeroMeanOneVar(X_pca);
X_pca = ZeroMeanOneVar(X_pca, x_mean,x_var);

%% Part 02: PCA Core
if dim<sample_num  % case: dim < sample_num
    St = X_pca * X_pca';
    [Vt, Dt, v] = svd(St);
    eig_value_unsort = diag(Dt);
    [eig_value , eig_index]  = sort(eig_value_unsort,'descend');
    
    eig_num = dim;
    W_pca = zeros(dim,eig_num);  %column of W_pca are main componet
    for i = 1:eig_num
        W_pca(:, i) = Vt(:,eig_index(i));
    end
    
else        % case: dim > sample_num
    St = X_pca' * X_pca;
    [Vt, Dt, v] = svd(St);
    eig_value_unsort = diag(Dt);
    [eig_value , eig_index]  = sort(eig_value_unsort,'descend');

    eig_num = sample_num;
    W_pca = zeros(dim,eig_num);  %column of W_pca are main componets
    for i = 1:eig_num
        W_pca(:, i) = X_pca*Vt(:,eig_index(i)); % similar with kernel situation ,eigvector is the combination of X_pca;
    end
end
%% Part 03: noramlized
for i = 1:eig_num
    v = W_pca(:, i);
    v = v/norm(v);
    W_pca(:, i) = v;
end
%% Part 04: keep Energe
sumEig = sum(eig_value);
sumEig = sumEig*PCARatio;
sumNow = 0;
for idx = 1:length(eig_value)
    sumNow = sumNow + eig_value(idx);
    if sumNow >= sumEig
        break;
    end
end
W_pca = W_pca(:,1:idx);
%% Part 05: projection and return
Y_PCA =  W_pca' * X_pca;
num_beg = 1;
SY2 = cell(nsets,1);
for i = 1 : nsets
    Y1 = SY1{i};
    n = size(Y1,2);
    SY2{i} = Y_PCA(:,num_beg:num_beg+n-1);    
    num_beg = num_beg + n;
end
