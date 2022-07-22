function w = gradientDescent_squaredError(z,y,w0,gradDescentParameters)

N = size(z,2); % total number of samples

if isequal(gradDescentParameters.type,'batch')

    alpha = gradDescentParameters.stepSize;
    epsilon = gradDescentParameters.stoppingCriterionThreshold;
    minIterCount = gradDescentParameters.minIterCount;
    wtrue = gradDescentParameters.wtrue;
    
    w = w0; yEstimated = w'*z;
    gradient = -2/N*sum((y-yEstimated).*z,2); 
    iterCounter = 0;
    while iterCounter < minIterCount || norm(gradient) > epsilon
        [iterCounter,norm(gradient)],
        if size(w,1)==2
            figure(1), 
            plot(w(1),w(2),'.'), axis equal, hold on,
            plot(wtrue(1),wtrue(2),'+r'),axis equal,
            xlabel('w_1'), ylabel('w_2'),
            title('Linear Case: True Weight Vector (+) and Estimated Weights At Each Iteration (.)'),
            drawnow,
        end 
        % Perform line search to select alpha satisfying Wolfe conditions
        w = w - alpha*gradient;
        yEstimated = w'*z;
        gradient = -2/N*sum((y-yEstimated).*z,2);
        iterCounter = iterCounter + 1;
    end
    
elseif isequal(gradDescentParameters.type,'stochastic')
    
    alpha = gradDescentParameters.stepSize;
    epsilon = gradDescentParameters.stoppingCriterionThreshold;
    minIterCount = gradDescentParameters.minIterCount;
    Nmb = gradDescentParameters.miniBatchSize;
    wtrue = gradDescentParameters.wtrue;
    
    w = w0; 
    ind = randi([1,N],1,Nmb); % pick mini batch samples uniform-randomly with replacement
    yEstimated = w'*z(:,ind);
    gradient = -2/Nmb*sum((y(ind)-yEstimated).*z(:,ind),2);
    averageNormGradient = norm(gradient);
    iterCounter = 0; averageNormGradient = norm(gradient);
    while iterCounter < minIterCount || averageNormGradient > epsilon
        [iterCounter,averageNormGradient],
        if size(w,1)==2
            figure(2), plot(w(1),w(2),'.'), axis equal, hold on,
            plot(wtrue(1),wtrue(2),'+r'),axis equal,
            xlabel('w_1'), ylabel('w_2'),
            title('Linear Case-stochastic GD: True Weight Vector (+) and Estimated Weights At Each Iteration '),
            drawnow,
        end
        w = w-alpha*gradient;
        % Prepare for next iteration
        ind = randi([1,N],1,Nmb); % pick mini batch samples uniform-randomly with replacement
        yEstimated = w'*z(:,ind);
        gradient = -2/Nmb*sum((y(ind)-yEstimated).*z(:,ind),2); 
        averageNormGradient = 0.9*averageNormGradient + 0.1*norm(gradient);
        iterCounter = iterCounter + 1;
    end 



end 
