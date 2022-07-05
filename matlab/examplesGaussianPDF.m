clc, clear all, close all

N=100; 

%% 1D case 
pdfParameters.Mean=5;
pdfParameters.Cov=4;
pdfParameters.type='Gaussian';
n = size(pdfParameters.Mean,1); % Determine data dimensionality from parameters
% Generate N points of random samples from pdf N(pdfParameters.Mean,pdfParameters.Cov))
x_1D=generateRandomSamples(N,n,pdfParameters,1);
% x~N(pdfParameters.Mean,pdfParameters.Cov)

%% 2D case 
clear pdfParameters 
pdfParameters.Mean=[5;1];
pdfParameters.Cov=[1 -1.5; -1.5 3];
pdfParameters.type='Gaussian';
n = size(pdfParameters.Mean,1); % Determine data dimensionality from parameters
% Generate N points of random samples from pdf N(pdfParameters.Mean,pdfParameters.Cov))
x_2D=generateRandomSamples(N,n,pdfParameters,1);

%% 3D case 
clear pdfParameters 
pdfParameters.Mean=[1;-10;3];%select mean of x,y and z axis distributions;
pdfParameters.Cov=[2 -.5 -.4; -.5 6 -.2; -.4 -.2 10 ];%select covariance of data
pdfParameters.type='Gaussian';
n = size(pdfParameters.Mean,1); % Determine data dimensionality from parameters
% Generate N points of random samples from pdf N(pdfParameters.Mean,pdfParameters.Cov))
x_3D=generateRandomSamples(N,n,pdfParameters,1);
