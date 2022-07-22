function [pEminERM,pEminLDA] = Exam1_Q3(N)
% Implements two classifiers (MAP and FisherLDA) and evaluates their
% ROC curves using 2-dim samples drawn form Gaussian distributions.

% Input N is the total number of samples drawn.
out = NaN; % Dummy output as a place holder.
close all,

n = 2; % number of feature dimensions
mu(:,1) = [-1;0]; mu(:,2) = [1;0];
Sigma(:,:,1) = [16 0;0 1]; Sigma(:,:,2) = [1 0;0 16];
p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % reserve space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
end
% Display samples from both classes
figure(1), subplot(2,2,1),
plot(x(1,label==0),x(2,label==0),'bo'), hold on,
plot(x(1,label==1),x(2,label==1),'k+'), axis equal,
title('Data and Their Class Labels'),

% Expected Risk Minimization Classifier (using true model parameters)
% In practice the parameters would be estimated from training samples
% Using log-likelihood-ratio as the discriminant score for ERM
discriminantScoreERM = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);

% MAP classifier (is a special case of ERM corresponding to 0-1 loss)
lambdaMAP = [0 1;1 0]; % 0-1 loss values yield MAP decision rule
gammaMAP = (lambdaMAP(2,1)-lambdaMAP(1,1))/(lambdaMAP(1,2)-lambdaMAP(2,2)) * p(1)/p(2); %threshold for MAP
decisionMAP = (discriminantScoreERM >= log(gammaMAP));
ind00MAP = find(decisionMAP==0 & label==0); p00MAP = length(ind00MAP)/Nc(1); % probability of true negative
ind10MAP = find(decisionMAP==1 & label==0); p10MAP = length(ind10MAP)/Nc(1); % probability of false positive
ind01MAP = find(decisionMAP==0 & label==1); p01MAP = length(ind01MAP)/Nc(2); % probability of false negative
ind11MAP = find(decisionMAP==1 & label==1); p11MAP = length(ind11MAP)/Nc(2); % probability of true positive
pEminERM = [p10MAP,p01MAP]*Nc'/N; % probability of error for MAP classifier, empirically estimated
ROCMAP = [p10MAP;p11MAP]; % MAP classifier on ROC curve for ERM classifier family

% Display MAP decisions
figure(1), subplot(2,2,2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00MAP),x(2,ind00MAP),'og'); hold on,
plot(x(1,ind10MAP),x(2,ind10MAP),'or'); hold on,
plot(x(1,ind01MAP),x(2,ind01MAP),'+r'); hold on,
plot(x(1,ind11MAP),x(2,ind11MAP),'+g'); hold on,
axis equal,
title('MAP Decisions (RED incorrect)');
% Construct the ROC for expected risk minimization by changing log(gamma)
[ROCERM,~] = estimateROC(discriminantScoreERM,label);
% Display the estimated ROC curve for ERM and indicate MAP on it
figure(1), subplot(2,2,3),
plot(ROCERM(1,:),ROCERM(2,:),'b'); hold on,
plot(ROCMAP(1),ROCMAP(2),'r.');

% Fisher LDA Classifer (using true model parameters)
% In practice the parameters would be estimated from training samples
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
% equivalently alpha w  = inv(Sw) Sb w
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
discriminantScoreLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
% Estimate the ROC curve for this LDA classifier
[ROCLDA,tauLDA] = estimateROC(discriminantScoreLDA,label);
probErrorLDA = [ROCLDA(1,:)',1-ROCLDA(2,:)']*Nc'/N; % probability of error for LDA for different threshold values
pEminLDA = min(probErrorLDA); ind = find(probErrorLDA == pEminLDA);
% Display the estimated ROC curve for LDA and indicate the operating points
% with smallest empirical error probability estimates (could be multiple)
figure(1), subplot(2,2,3),
plot(ROCLDA(1,:),ROCLDA(2,:),'b:'); hold on,
plot(ROCLDA(1,ind),ROCLDA(2,ind),'r.'); 
axis equal, xlim([0,1]); ylim([0,1]);
title('ROC Curves for ERM and LDA'),
decisionLDA = (discriminantScoreLDA >= tauLDA(ind(1))); % use smallest min-error threshold
ind00LDA = find(decisionLDA==0 & label==0); p00LDA = length(ind00LDA)/Nc(1); % probability of true negative
ind10LDA = find(decisionLDA==1 & label==0); p10LDA = length(ind10LDA)/Nc(1); % probability of false positive
ind01LDA = find(decisionLDA==0 & label==1); p01LDA = length(ind01LDA)/Nc(2); % probability of false negative
ind11LDA = find(decisionLDA==1 & label==1); p11LDA = length(ind11LDA)/Nc(2); % probability of true positive

% Display LDA decisions
figure(1), subplot(2,2,4), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00LDA),x(2,ind00LDA),'og'); hold on,
plot(x(1,ind10LDA),x(2,ind10LDA),'or'); hold on,
plot(x(1,ind01LDA),x(2,ind01LDA),'+r'); hold on,
plot(x(1,ind11LDA),x(2,ind11LDA),'+g'); hold on,
axis equal,
title('LDA Decisions (RED incorrect)');

disp(strcat('Smallest P(error) for ERM = ',num2str(pEminERM))),
disp(strcat('Smallest P(error) for LDA = ',num2str(pEminLDA))),

keyboard,

%%%
function [ROC,tau] = estimateROC(discriminantScore,label)
% Generate ROC curve samples
Nc = [length(find(label==0)),length(find(label==1))];
sortedScore = sort(discriminantScore,'ascend');
tau = [sortedScore(1)-1,(sortedScore(2:end)+sortedScore(1:end-1))/2,sortedScore(end)+1];
%thresholds at midpoints of consecutive scores in sorted list
for k = 1:length(tau)
    decision = (discriminantScore >= tau(k));
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
    ROC(:,k) = [p10;p11];
end

%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu); 
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);