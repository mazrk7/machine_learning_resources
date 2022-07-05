function x = generateRandomSamples(N,n,pdfParameters,visualizationFlag)
% Generates N vector valued samples with dimensionality n according to the
% probability density function specified by pdfParameters.
% pdfParameters include type of pdf and relevant parameters
% Data will be visualized if visualizationFlag==1 and 0<n<3

if isequal(pdfParameters.type , 'Gaussian')
    % For a Gaussian pdf 
    % pdfParameters.Mean is = Mean Vector
    % pdfParameters.Cov is = CovMatrix
    [V,D] = eig(pdfParameters.Cov);
    pdfParameters.Scale = V*D^0.5;
    
    z = randn(n,N); 
    % z~N(0,I) are zero-mean identity-covariance Gaussian samples
    x = pdfParameters.Scale*z+pdfParameters.Mean;
    % x~N(pdfParameters.Mean,pdfParameters.Cov)
   
elseif isequal(pdfParameters.type , 'Uniform')
    % For a Uniform pdf 
    % pdfParameters.Mean is = Mean Vector
    % pdfParameters.Scale is a matrix that skews the volume to a parallelogram
    z = 2*(rand(n,N)-0.5); 
    % z~Uniform[-1,1]^n are zero-mean "unit-scale" uniformly distributed samples
    x = pdfParameters.Scale.*z+pdfParameters.Mean;
    % x~Uniform(pdfParameters.Mean,pdfParameters.Scale)
end

%plot if visualizationFlag is on
if visualizationFlag==1 & 0<n & n<=3
    figure
    if n==1
        subplot(1,2,1), plot(z,zeros(1,N),'.'); title('z ~ Standard Shift and Scale');
        subplot(1,2,2), plot(x,zeros(1,N),'.'); title('x ~ Specified Shift and Scale');
    elseif n==2
        subplot(1,2,1), plot(z(1,1:N),z(2,1:N),'.'); title('z ~ Standard Shift and Scale');
        subplot(1,2,2), plot(x(1,1:N),x(2,1:N),'.'); title('x ~ Specified Shift and Scale');
    elseif n==3
        subplot(1,2,1), plot3(z(1,:),z(2,:),z(3,:),'.'); title('z ~ Standard Shift and Scale');
        subplot(1,2,2), plot3(x(1,:),x(2,:),x(3,:),'.'); title('x ~ Specified Shift and Scale');
    end
end
 