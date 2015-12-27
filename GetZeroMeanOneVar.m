function [x_mean x_var] = GetZeroMeanOneVar(X)
Y = double(X);
x_mean = mean(Y,2); %mean rows
x_var = std(Y,0,2);