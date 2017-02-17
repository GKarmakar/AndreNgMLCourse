function [J, grad] = costfunction(theta, X, y)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
X = [ones(m, 1) X];
h = (X * theta);
error = (h - y);
sum_error_sqr = sum(error.^2);
J = (1/(2*m) * sum_error_sqr);
v = sum(X' * error);
grad = v ./ m;
grad = grad(:);
% =========================================================================

end
