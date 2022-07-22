function [cost,h,gradient,z] = binaryCrossEntropyCostFunction(w,x,labels,ModelType)
N = size(x,2);
% Cost function to be minimized to optimize theta
[h,z]=logisticGeneralizedLinearModel(x,w,ModelType); % logistic function (sigmoid type nonlinearity)
cost = (-1/N)*sum(labels.*log(h)+(1-labels).*log(1-h));
gradient = z*(h-labels)'/N; % gradient of the cost  function with respect to the wieghts w 
end 