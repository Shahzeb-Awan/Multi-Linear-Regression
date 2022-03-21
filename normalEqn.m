function [theta] = normalEqn(X, y)
%Computes the closed-form solution to linear regression 
theta = zeros(5, 1);
theta=pinv(X'*X)*X'*y;
end
