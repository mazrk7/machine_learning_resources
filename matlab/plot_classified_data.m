function error=plot_classified_data(decision, label,Nc,p,fig,x,w,type,n)
%Plots Incorrect and correct decisions (and boundary) based on original
%class labels 

%Find all correct and incorrect decisions 
TN=find(decision==0&label==0);%True negative
FP=find(decision==1&label==0); pFA=length(FP)/Nc(1);%false positive
FN=find(decision==0&label==1); pMD=length(FN)/Nc(2); % false negative
TP= find(decision==1& label==1); %true positive
error=(pFA*p(1)+pMD*(p(2)))*100; % calculate total error

% Plot all decisions (green=correct, red= incorrect)
subplot(fig(1),fig(2),fig(3));
%plot decisions for n=2
if n==2 
plot(x(2,TN),x(3,TN),'og');hold on ; 
plot(x(2,FP),x(3,FP),'or');hold on ;
plot(x(2,FN),x(3,FN),'+r');hold on ;
plot(x(2,TP),x(3,TP),'+g');hold on ;
%Plot boundary based on whether its linear(L) or non-linear(Q)
if type =='L'
    %Plot straight line if boundary is linear
    boundX=[min(x(:,2))-2, max(x(:,2))+2];
    boundY=(-1./w(3)).*(w(2).*boundX+w(1));
    plot(boundX,boundY);
elseif type=='Q'
    x_grid=linspace(min(x(:,2))-2, max(x(:,2))+2);
    y_grid=linspace(min(x(:,3))-2, max(x(:,3))+2);
    
    score= get_boundary(x_grid,y_grid, w);
    contour (x_grid, y_grid, score,[0,0]); 
end 
%plot decisions for n=3
elseif n==3 
plot3(x(2,TN),x(3,TN),x(4,TN),'og');hold on ; 
plot3(x(2,FP),x(3,FP),x(4,FP),'or');hold on ;
plot3(x(2,FN),x(3,FN),x(4,FN),'+r');hold on ;
plot3(x(2,TP),x(3,TP),x(4,TP),'+g');hold on ;
end 


legend( 'Class 0 Correct Decisions' , 'Class 0 Wrong Decisions',...
    'Class 1 Wrong Decisions', 'Class 1 Correct Decisions', 'Classifier','location','north');
xlabel('x_1');ylabel('x_2');
end 