clc, clear all , close all,

N=100; % Specify number of samples

%% 1D case
% Store parameters in a structure
pdfParameters.a=[1];%uniform distribution lower endpoint of the x axis
pdfParameters.b=[5 ] ; %uniform distribution higher endpoint of the x axis
pdfParameters.Scale=(pdfParameters.b-pdfParameters.a)/2;
pdfParameters.Mean=(pdfParameters.a+pdfParameters.b)/2;
pdfParameters.type='Uniform';
n = size(pdfParameters.a,1); % Determine data dimensionality from pdf parameters

% Generate N points of random samples from pdf Uniform(pdfParameters.Mean,pdfParameters.Scale)
x_1D=generateRandomSamples(N,n,pdfParameters,1);

clear pdfParameters 
%% 2D case
% Store parameters in a structure
pdfParameters.a=[1 ;5];%uniform distribution lower endpoints of the x and y axis
pdfParameters.b=[5; 10 ] ; %uniform distributions higher endpoint of the x and y axis
pdfParameters.Scale=(pdfParameters.b-pdfParameters.a)/2;
pdfParameters.Mean=(pdfParameters.a+pdfParameters.b)/2;
pdfParameters.type='Uniform';
n = size(pdfParameters.a,1); % Determine data dimensionality from pdf parameters

% Generate N points of random samples from pdf Uniform(pdfParameters.Mean,pdfParameters.Scale)
x_2D=generateRandomSamples(N,n,pdfParameters,1);

clear pdfParameters
%% 3D case
% Store parameters in a structure
pdfParameters.a=[1 ;5; -3];%uniform distribution lower endpoints of the x, y and z axis
pdfParameters.b=[5; 10 ; -10] ; %uniform distributions higher endpoint of the x,y and z axis
pdfParameters.Scale=(pdfParameters.b-pdfParameters.a)/2;
pdfParameters.Mean=(pdfParameters.a+pdfParameters.b)/2;
pdfParameters.type='Uniform';
n = size(pdfParameters.a,1); % Determine data dimensionality from pdf parameters

% Generate N points of random samples from pdf Uniform(pdfParameters.Mean,pdfParameters.Scale)
x_3D=generateRandomSamples(N,n,pdfParameters,1);

