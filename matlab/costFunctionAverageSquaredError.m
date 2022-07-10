function cost = costFunctionAverageSquaredError(z,y,w)

% Input vector z, target output y...
N = size(z,2);


% Cost function to be minimized to optimize w
yEstimate = w'*z;
cost = (1/N)*sum(sum((y-yEstimate).^2,1),2);
end 