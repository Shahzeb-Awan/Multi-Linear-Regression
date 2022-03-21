function [X_norm, mu, sigma] = featureNormalize(X)
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 
X_norm = X;
mu = zeros(1, 5);
sigma = zeros(1, 5);
 
mu=mean(X);
sigma=std(X);
X_norm=(X-mu)./sigma;
end
