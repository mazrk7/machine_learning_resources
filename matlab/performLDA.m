function [w,y]=performLDA(x,labels)
% Performs Fisher Linear Discriminant Analysis on data from two classes.
% Receives real-vector-valued samples in the columns of x such that x has
% size nxN where n is the sample dimensionality and N is the sample count.
% Class labels corresponding to each sample is received in the 1xN row
% vector, second input argumant labels.
% Returns nx1 w, which is the Fisher LDA projection vector, and 1xN y,
% which are the scalar LDA projections of the input samples.
% Note that, one could use multiple top generalized eigenvectors and not
% just the one with the largest generalized eigenvector, and obtain a
% matrix W to get multidimensional projection y=W'*x and obtain a
% dimensionality reduction solution that attempts to maintain class
% separability in y with more than 1 dimension. This would be a crude but
% quick way to achieve class-separability-preserving linear dimensionality
% reduction, using the Fisher LDA objective as a measure of class
% separability.

% Estimate mean vectors and covariance matrices from samples
mu(:,1) = mean(x(:,find(labels==1)),2); S(:,:,1) = cov(x(:,find(labels==1))');
mu(:,2) = mean(x(:,find(labels==2)),2); S(:,:,2) = cov(x(:,find(labels==2))');

% Calculate the between/within-class scatter matrices
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = S(:,:,1) + S(:,:,2);

% Solve for the Fisher LDA projection vector (in w)
[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector
% W = V(:,ind(1:m)) 
% For a specified m between 1 and n the previous line will determing a
% multidimensional projection matrix. Needs to receive m as an input and
% return W as output, if you want this capability.
y = w'*x 
end 

