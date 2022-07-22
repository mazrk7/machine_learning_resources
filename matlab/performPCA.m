function [Q,D,xzm,yzm]=performPCA(x)
% Performs PCA on real-vector-valued data.
% Receives input x with size nxN where n is the dimensionality of samples
% and N is the number of vector-valued samples.
% Returns Q, which is an orthogonal matrix that contains in its columns the
% PCA projection vectors ordered from first to last; D, which is a diagonal
% matrix that contains the variance of each principal component
% corresponding to these projection vectors. Zero-mean version of the
% samples and zero-mean principal component projections are also returned
% in xzm and yzm, respectively. 

[n,N]=size(x);
% Sample-based estimates of mean vector and covariance matrix
muhat = mean(x,2); % Estimate the mean vector using the samples
Sigmahat = cov(x'); % Estimate the covariance matrix using the samples
% Subtract the estimated mean vector to make the data 0-mean
xzm = x - muhat*ones(1,N); % Obtain zero-mean sample set
% Get the eigenvectors (in Q) and eigenvalues (in D) of the
% estimated covariance matrix
[Q,D] = eig(Sigmahat);
% Sort the eigenvalues from large to small, reorder eigenvectors
% accordingly as well.
[d,ind] = sort(diag(D),'descend');
Q = Q(:,ind); D = diag(d);
yzm = Q'*xzm; % Principal components of x (zero-mean)
end 

