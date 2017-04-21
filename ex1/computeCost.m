function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% You need to return the following variables correctly 
J = 0;

% Initialize some useful values
m = length(y); % number of training examples


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
y_ = X*theta;
J = sum(sum( (y_-y).^2))/(2*m);





% =========================================================================

end
