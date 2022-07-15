% Expected risk minimization with C classes
clear all, close all,

% Indicate if you want visualization plots
visualizationFlag = 1;% 0: no plots, 1: visualization plots

C = 3; % Number of classes
M = 7; % Number of Gaussian components
N = 1000; % Number of samples
n = 2; % Data dimensionality (must be 2 for plots to work)

% Specify loss-matrix for ERM classifier design
lossMatrix = ones(C,C)-eye(C); % Using 0-1 loss to minimize probability-of-error

% Specify the data pdf
gmmParameters.component2label = [1,1,2,2,3,3,3]; % This row vector indicates which Gaussian component belongs to which class
gmmParameters.priors = ones(1,M)/M; % uniform component priors
gmmParameters.meanVectors = repmat(2*gmmParameters.component2label,n,1)+1.25*n*M*rand(n,M); % arbitrary mean vectors
gmmParameters.numberOfClasses = C;
for m = 1:M
    A = eye(n)+0.2*randn(n,n);
    gmmParameters.covMatrices(:,:,m) = A'*A; % arbitrary covariance matrices
end
% Generate samples from specified data pdf
[x,componentLabels] = generateDataFromGMM(N,gmmParameters,0); % Generate data
% Convert component labels to class labels for each sample
labels = zeros(1,N);
for m = 1:M
    indm = find(componentLabels == m);
    labels(1,indm) = repmat(gmmParameters.component2label(m),1,length(indm));
end
% Determine ERM decisions for each samples using specified (true) data PDF
[decisions,confusionMatrix] = performERMclassificationGMMdata(x,labels,gmmParameters,lossMatrix,visualizationFlag);

% Second example: C=2, p(x|L=1) has GMM with 2 components, p(x|L=2) has GMM
% with 3 components