function [w,z] = gradientDescent_binaryCrossEntropy(x,labels,gradDescentParameters)

N = size(x,2); % total number of samples

% Training weights using gradient and stochastic gradient descents
if isequal(gradDescentParameters.type,'batch')

    alpha = gradDescentParameters.stepSize;
    epsilon = gradDescentParameters.stoppingCriterionThreshold;
    minIterCount = gradDescentParameters.minIterCount;
    ModelType=gradDescentParameters.ModelType;
    
    % Initialize estimates for weights
    if isequal(ModelType,'logisticLinear')
        w = 6*randn(3,1);  
    elseif isequal(ModelType,'logisticQuadratic')
        w = 6*randn(7,1); 
    end
    
    [cost,h,gradient,z] = binaryCrossEntropyCostFunction(w,x,labels,ModelType);
    iterCounter = 0;
    %perform gradient descent
    while iterCounter < minIterCount || norm(gradient) > epsilon
        
        w = w - alpha*gradient;
        [cost,h,gradient,z] = binaryCrossEntropyCostFunction(w,x,labels,ModelType);
        iterCounter = iterCounter + 1;
    end
    
elseif isequal(gradDescentParameters.type,'stochastic')
    
    alpha = gradDescentParameters.stepSize;
    epsilon = gradDescentParameters.stoppingCriterionThreshold;
    minIterCount = gradDescentParameters.minIterCount;
    ModelType=gradDescentParameters.ModelType;
    Nmb = gradDescentParameters.miniBatchSize;
        
    %Initialize estimates for weights
    if isequal(ModelType,'logisticLinear')
        w = 6*randn(3,1); 
    elseif isequal(ModelType,'logisticQuadratic')
        w = 6*randn(7,1);
    end 
    
    ind = randi([1,N],1,Nmb); % pick mini batch samples uniform-randomly with replacement
    [cost,h,gradient,z] = binaryCrossEntropyCostFunction(w,x(:,ind),labels(ind),ModelType); 
    iterCounter = 0; averageNormGradient = norm(gradient);
    %perform gradient descent
    while iterCounter < minIterCount || averageNormGradient > epsilon
 
        w = w-alpha*gradient;
        [cost,h,gradient,z] = binaryCrossEntropyCostFunction(w,x(:,ind),labels(ind),ModelType);
        averageNormGradient = 0.9*averageNormGradient + 0.1*norm(gradient);
        iterCounter = iterCounter + 1;
    end 



end 
