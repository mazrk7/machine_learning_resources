function plotFunctionContoursOnData(x,functionSpecs,figureLabel)

% Create the grid in 2-dim data space to cover the data range
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
if isequal(functionsSpecs.type,'logisticLinear')
    discriminantScoreGridValues = logisticGeneralizedLinearModel(x,functionsSpecs.w,functionsSpecs.type);
    %minDSGV = min(discriminantScoreGridValues);
    %maxDSGV = max(discriminantScoreGridValues);
    contourLevels = linspace(0,1,11); % contours at 10% increments over range of discriminant scores
    discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
    figure(figureLabel), 
    contour(horizontalGrid,verticalGrid,discriminantScoreGrid,contourLevels); % plot equilevel contours of the discriminant function 
    %legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
    title('Contours of logistic-linear model for P(L=1|x) as a function of 2-dimensional x'),
    xlabel('x_1'), ylabel('x_2'), 
elseif isequal(functionsSpecs.type,'logisticQuadratic')
    discriminantScoreGridValues = logisticGeneralizedLinearModel(x,functionsSpecs.w,functionsSpecs.type);
    %minDSGV = min(discriminantScoreGridValues);
    %maxDSGV = max(discriminantScoreGridValues);
    contourLevels = linspace(0,1,11); % contours at 10% increments over range of discriminant scores
    discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
    figure(figureLabel), 
    contour(horizontalGrid,verticalGrid,discriminantScoreGrid,contourLevels); % plot equilevel contours of the discriminant function 
    %legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
    title('Contours of logistic-quadratic model for P(L=1|x) as a function of 2-dimensional x'),
    xlabel('x_1'), ylabel('x_2'), 
end
end


