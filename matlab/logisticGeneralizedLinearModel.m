function [h,z] = logisticGeneralizedLinearModel(x,w,ModelType)

N = size(x,2); % determine size of data vectors
n=size(x,1); % determine dimensionality of data vectors
if isequal(ModelType,'logisticLinear')
    % Data augmentation for logistic-linear model
    z = [ones(1,N);x]; 
elseif isequal(ModelType,'logisticQuadratic')
    % Data augmentation for logistic-quadratic model
    z = [ones(1,N);x];
    for r = 1:n
        for c = 1:n
            z = [z;x(r,:).*x(c,:)];
        end
    end    
elseif isequal(ModelType,'none')
    z = x;
end

h = 1./(1+exp(-w'*z)); % logistic function (sigmoid type nonlinearity)