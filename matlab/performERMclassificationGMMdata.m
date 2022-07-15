function [decisions,confusionMatrix] = performERMclassificationGMMdata(x,labels,gmmParameters,lossMatrix,visualizationFlag)
% Performs ERM  on real-vector-valued data where each class conditional pdf is a GMM.
% Receives labeled input x with size nxN and labels labels where n is the
% dimensionality of samples and N is the number of vector-valued samples. 
% Receives Gaussian mixture parameters that were used to generate the data
% and the loss matrix that associates each data category with a loss value. 
% Returns ConfusionMatrix, which is a matrix that contains in its off diagonal 
% elements the percentage of times that the selected labels were not the
% true labels; decisions, which is a vector that contains the decision of
% each data element that minimizes the risk.

% Number of classes = C
C = gmmParameters.numberOfClasses;
for l = 1:C
    components{l} = find(gmmParameters.component2label==l);
    classPriors(l,1) = sum(gmmParameters.priors(components{l}));
    tempP = gmmParameters.priors(components{l})/classPriors(l,1);
    tempMU = gmmParameters.meanVectors(:,components{l})';
    tempSIGMA = gmmParameters.covMatrices(:,:,components{l});
    classConditionalPDF{l} = gmdistribution(tempMU,tempSIGMA,tempP);
end
[n,N] = size(x);
% Evaluate class conditional likelihoods p(x|L=l) for each sample-label pair.
for l = 1:C
    %pxgivenl(l,:) = evalGaussianPDF(x,gmmParameters.meanVectors(:,l),gmmParameters.covMatrices(:,:,l)); 
    pxgivenl(l,:) = pdf(classConditionalPDF{l},x');
end
%determine the overall data pdf
classConditionalPDFTotal = gmdistribution(gmmParameters.meanVectors',gmmParameters.covMatrices,gmmParameters.priors);

% Evaluate the likelihood of each sample according to the overall data pdf
%px=classPriors'*pxgivenl;
px = pdf(classConditionalPDFTotal,x')';%pdf(gmmParameters,x); % Note that gmmParameters is a mixture of class-conditional GMMs
% Evaluate the class posterior probabilities for each label-sample pair
classPosteriors = pxgivenl.*repmat(classPriors,1,N)./repmat(px,C,1); % P(L=l|x)
% Evaluate expected risk (loss) values for each decision-sample pair
expectedRisks = lossMatrix*classPosteriors; % Expected Risk for each label (rows) for each sample (columns)
% Determine minimum-expected-risk decisions for each sample
[~,decisions] = min(expectedRisks,[],1); 
% Note: minimum expected risk decision with 0-1 loss is MAP classification
% and minimizes probability of error

% Estimate the confusion matrix P(D=d|L=l) from samples for each
% decision-label pair
mShapes = 'ox+*.'; % Accomodates up to C=5
mColors = 'rkbmy';
if visualizationFlag==1 & 0<n & n<=3
    figure(1), clf,
end 
for d = 1:C % each decision option
    for l = 1:C % each class label
        ind_dl = find(decisions==d & labels==l);
        confusionMatrix(d,l) = length(ind_dl)/length(find(labels==l));
        if visualizationFlag==1 & 0<n & n<=3
            figure(1), 
            plot(x(1,ind_dl),x(2,ind_dl),strcat(mShapes(l),mColors(d))), 
            hold on, axis equal,
            title('True label = Marker shape; Decision = Marker Color')
            if d~=l
                plot(x(1,ind_dl),x(2,ind_dl),strcat(mShapes(l),mColors(d)),'Markersize',16),
            end
        end
    end
end

end
    
