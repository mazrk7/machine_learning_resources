clear all; close all; clc 

%%% In this example,we train a classifier using gradient and stochastic gradient
%%% descents using data from 10000 samples respectively.first,in each case, 
%%% validation and training data are generated using 2D and 3D Gaussian mixtures.
%%% classifiers are trained using gradient and stochastic gradient descents and 
%%% trained weights are generated. these weights are used after to classify the 
%%% validation data. Classification results are plotted for the 2D and 3D case
%%% respectively, 

%% 2D case
% set-up: givene parameters and validation data
% given parameters

nx = 1; % data dimensionality for input x
ny = 1; % data dimensionality for output y
n = nx + ny; %total data dimensionality
NTrain = 100; % Specify number of training samples for experiments
NVal = 10000; % Specify number of validation samples for experiments
epsilon = 1.; % stopping criterion threshold/tolerance
alpha = 0.01; % step size for gradient descent methods

% Specify data pdf
gmmParameters.priors = [0.3,0.4,0.3] ; % class priors
gmmParameters.meanVectors = [-5 0 5;-2 0 2];
gmmParameters.covMatrices(:,:,1) = [2 0;0 0.1]; 
gmmParameters.covMatrices(:,:,2) = [1 0.9;0.9 1];
gmmParameters.covMatrices(:,:,3) = [2 0;0 0.1]; 

% Generate iid training and validation samples
[dataTrain,~] = generateDataFromGMM(NTrain,gmmParameters,0);
[dataVal,~] = generateDataFromGMM(NVal,gmmParameters,0);
% Specify input/output data pairs
xTrain = dataTrain(1:nx,:); yTrain = dataTrain(nx+1:n,:); % Model will estimate y from x
xVal = dataVal(1:nx,:); yVal = dataVal(nx+1:n,:); % Model will estimate y from x

%% Linear 
   
% Data augmentation for logistic-linear model
zTrainLin = [ones(1,NTrain);xTrain]; 
zValLin = [ones(1,NVal);xVal]; 

% Analytical solution
wAnalyticalLin = inv(zTrainLin*zTrainLin')*(zTrainLin*yTrain'); % omitting the 1/N factors in both terms

% Shared initial weights for iterative gradient descent methods...
w0 = 6*randn(size(zTrainLin,1),1); % Initial estimates for weights  

% Deterministic (batch) gradient descent 
% Uses all samples in training set for each gradient calculation
paramsGD.type = 'batch';
paramsGD.stepSize = alpha;
paramsGD.stoppingCriterionThreshold = epsilon;
paramsGD.minIterCount = 10;
paramsGD.wtrue = wAnalyticalLin;
wGradDescentLin = gradientDescent_squaredError(zTrainLin,yTrain,w0,paramsGD);

% Stochastic (with mini-batch) gradient descent
% Uses randomly picked samples to estimate gradient
paramsSGD.type = 'stochastic';
paramsSGD.stepSize = alpha;
paramsSGD.stoppingCriterionThreshold = epsilon;
paramsSGD.minIterCount = 10;
paramsSGD.miniBatchSize = 10; % 
paramsSGD.wtrue = wAnalyticalLin;
wStochasticGradDescentLin = gradientDescent_squaredError(zTrainLin,yTrain,w0,paramsSGD);


% Linear model estimates for y using analytical weights
yhatTrain = wAnalyticalLin'*zTrainLin;
yhatVal = wAnalyticalLin'*zValLin;

% Linear model estimates for y using  gradient descent weigths
yhatTrainGD=wGradDescentLin'*zTrainLin;
yhatValGD=wGradDescentLin'*zValLin;

yhatTrainSGD=wStochasticGradDescentLin'*zTrainLin;
yhatValSGD=wStochasticGradDescentLin'*zValLin;


% Linear model estimates for y using fmin search weigths
[wFmin,~]= fminsearch(@(w)(costFunctionAverageSquaredError(zTrainLin,yTrain,w)),w0);
yhatTrainFmin=wFmin'*zTrainLin;
yhatValFmin=wFmin'*zValLin;

figure(3), clf,
subplot(1,2,1), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrain,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Training Data'),
subplot(1,2,2), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatVal,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Validation Data'),

figure(4)
subplot(3,2,1), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrainGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Training Data  GD'),
subplot(3,2,2), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatValGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Validation Data GD'),
subplot(3,2,3), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrainSGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Training Data SGD'),
subplot(3,2,4), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatValSGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Validation Data SGD'),
subplot(3,2,5), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrainFmin,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Training Data Fmin search'),
subplot(3,2,6), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatValFmin,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Linear Validation Data Fmin search')
%% Quadratic 
clear w0;
zTrainQuad = [ones(1,NTrain);xTrain];
    for r = 1:nx
        for c = 1:nx
            zTrainQuad = [zTrainQuad;xTrain(r,:).*xTrain(c,:)];
        end
    end
    
    zValQuad = [ones(1,NVal);xVal];
    for r = 1:nx
        for c = 1:nx
            zValQuad = [zValQuad;xVal(r,:).*xVal(c,:)];
        end
    end
    
% Analytical solution
wAnalyticalQuad = inv(zTrainQuad*zTrainQuad')*(zTrainQuad*yTrain'); % omitting the 1/N factors in both terms

% Shared initial weights for iterative gradient descent methods...
w0 = 6*randn(size(zTrainQuad,1),1); % Initial estimates for weights  

% Deterministic (batch) gradient descent 
% Uses all samples in training set for each gradient calculation
paramsGD.type = 'batch';
paramsGD.stepSize = 0.05*alpha;
paramsGD.stoppingCriterionThreshold = epsilon;
paramsGD.minIterCount = 10;
wGradDescentQuad = gradientDescent_squaredError(zTrainQuad,yTrain,w0,paramsGD);

% Stochastic (with mini-batch) gradient descent
% Uses randomly picked samples to estimate gradient
paramsSGD.type = 'stochastic';
paramsSGD.stepSize = 0.05*alpha;
paramsSGD.stoppingCriterionThreshold = epsilon;
paramsSGD.minIterCount = 10;
paramsSGD.miniBatchSize = 10; % 
wStochasticGradDescentQuad = gradientDescent_squaredError(zTrainQuad,yTrain,w0,paramsSGD);

% Quadratic model estimates for y using analytical weights
yhatTrainQuad = wAnalyticalQuad'*zTrainQuad;
yhatValQuad = wAnalyticalQuad'*zValQuad;

% Quadratic model estimates for y using  gradient descent weigths
yhatTrainQuadGD=wGradDescentQuad'*zTrainQuad;
yhatValQuadGD=wGradDescentQuad'*zValQuad;

yhatTrainQuadSGD=wStochasticGradDescentQuad'*zTrainQuad;
yhatValQuadSGD=wStochasticGradDescentQuad'*zValQuad;

% Quadratic model estimates for y using fmin search weigths
[wFminQuad,~]= fminsearch(@(w)(costFunctionAverageSquaredError(zTrainQuad,yTrain,w)),w0);
yhatTrainQuadFmin=wFminQuad'*zTrainQuad;
yhatValQuadFmin=wFminQuad'*zValQuad;

figure(5), clf,
subplot(1,2,1), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrainQuad,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Training Data'),
subplot(1,2,2), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatValQuad,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Validation Data'),

figure(6), clf,
subplot(3,2,1), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrainQuadGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Training Data GD'),
subplot(3,2,2), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatValQuadGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Validation Data GD'),
subplot(3,2,3), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrainQuadSGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Training Data GDS'),
subplot(3,2,4), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatValQuadSGD,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Validation Data GDS'),
subplot(3,2,5), plot(xTrain,yTrain,'.k'), hold on, plot(xTrain,yhatTrainQuadFmin,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Training Data Fmin search'),
subplot(3,2,6), plot(xVal,yVal,'.k'), hold on, plot(xVal,yhatValQuadFmin,'.b');
axis equal, xlabel('x'), ylabel('y (black) or yhat (blue)'), title('Quadratic Validation Data Fmin search')
