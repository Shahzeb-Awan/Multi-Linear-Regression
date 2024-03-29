function [theta, J_history] = gradientDescentMulti2(X, y, theta, alpha, num_iters)
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters
% Perform a single gradient step on the parameter vector theta. 
gradient=zeros(6,1);
  for i=1:m,
    for j=1:6,
    gradient(j,1)=gradient(j,1)+(theta'*X(i,:)'-y(i))*X(i,j);
    end
  end
theta=theta-alpha/m*gradient;
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end
end