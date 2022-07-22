%% EECE5644 Summer 1 2022 - Exam 0 - Question 1
clear all; close all; clc;

m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1), clf, 
markerList = {'d','+','.'};
colorList = {'r','b','g'};
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),[markerList{l},colorList{l}]);
    axis equal; hold on; grid on; box on;
end
set(gca,'FontSize',16);
xlabel('Feature 1','FontSize',15); ylabel('Feature 2','FontSize',15);
title('Generated original data samples','FontSize', 18);
legend('True Class 1','True Class 2','True Class 3');

%% Evaluate the MPE discriminant function for each class and observation
[n,N] = size(x);
g = zeros(3,N); % will construct a discriminant scores matrix
for k = [1,2,3]
    mu = m(:,k);
    cov = Sigma(:,:,k);
    prior = classPriors(k);
    C = ((2*pi)^n * det(cov))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(cov)*(x-repmat(mu,1,N))),1);
    g(k,:) = log(C*exp(E)) + log(prior);
end
[g_k,k_hat] = max(g);

%% Present the results asked for...
Nc1 = sum(L==1); Nc2 = sum(L==2); Nc3 = sum(L==3);
fprintf('Number of samples from Class 1: %d, Class 2: %d, Class 3: %d \n', Nc1, Nc2, Nc3);
fprintf('Confusion Matrix (rows: Predicted class, columns: True class): \n');
confMat = confusionmat(k_hat,L); disp(confMat);
fprintf('Total number of misclassified samples: %d \n',  N - sum(diag(confMat)));
Pe = [(confMat(2,1)+confMat(3,1))/Nc1, (confMat(1,2)+confMat(3,2))/Nc2, (confMat(1,3)+confMat(2,3))/Nc3] * [Nc1, Nc2, Nc3]' / N; 
fprintf('Empirically Estimated Probability of Error: %.4f \n', Pe);

figure(2); clf,
t1p1 = ((L==1) & (k_hat==1)); t1p2 = ((L==1) & (k_hat==2)); t1p3 = ((L==1) & (k_hat==3));
t2p1 = ((L==2) & (k_hat==1)); t2p2 = ((L==2) & (k_hat==2)); t2p3 = ((L==2) & (k_hat==3));
t3p1 = ((L==3) & (k_hat==1)); t3p2 = ((L==3) & (k_hat==2)); t3p3 = ((L==3) & (k_hat==3));
axis equal; hold on; grid on; box on;
scatter(x(1,t1p1),x(2,t1p1),[markerList{1},colorList{1}]); 
scatter(x(1,t2p2),x(2,t2p2),[markerList{2},colorList{2}]);
scatter(x(1,t3p3),x(2,t3p3),[markerList{3},colorList{3}]); 
scatter(x(1,t1p2),x(2,t1p2),[markerList{2},colorList{1}]); 
scatter(x(1,t1p3),x(2,t1p3),[markerList{3},colorList{1}]); 
scatter(x(1,t2p1),x(2,t2p1),[markerList{1},colorList{2}]);
scatter(x(1,t2p3),x(2,t2p3),[markerList{3},colorList{2}]); 
scatter(x(1,t3p1),x(2,t3p1),[markerList{1},colorList{3}]); 
scatter(x(1,t3p2),x(2,t3p2),[markerList{2},colorList{3}]);
set(gca,'FontSize',16);
xlabel('Feature 1','FontSize',15); ylabel('Feature 2','FontSize',15);
title('Classification Decisions: marker shape for predicted labels, color for true labels','FontSize', 18);

% Generate an informative artifical legend with dummy data
dummy(1) = plot(nan,nan,[markerList{1},colorList{1}]);
dummy(2) = plot(nan,nan,[markerList{2},colorList{2}]);
dummy(3) = plot(nan,nan,[markerList{3},colorList{3}]);
dummy(4) = plot(nan,nan,[markerList{1},'k']);
dummy(5) = plot(nan,nan,[markerList{2},'k']);
dummy(6) = plot(nan,nan,[markerList{3},'k']);
legend(dummy, {'True Class 1','True Class 2','True Class 3','Predicted as Class 1', 'Predicted as Class 2', 'Predicted as Class 3'});
