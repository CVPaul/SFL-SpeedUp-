% 0-mean£¬1-covaraince£¬X is the data set whitch  contains n samples,
% input : x_mean(column vector)
%       : x_var(column vector)
function M = ZeroMeanOneVar(X,x_mean,x_var)
% index = find(x_var<0.0000001);
% x_var(index) = 1;

num = size(X,2);
X_mean = repmat(x_mean, 1, num);
X_var = repmat(x_var, 1, num);
M = X - X_mean;
M = M ./ X_var;

flag = isnan(M);
M(flag) = 0;