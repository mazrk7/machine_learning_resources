function plot_training_data(label,fig,x,w,type,n)
%Plots original class labels and decision boundary

subplot(fig(1),fig(2),fig(3));
if n==2
plot(x(2,label==0), x(3,label==0),'o');hold on 
plot(x(2,label==1), x(3,label==1),'+');
%Plot boundary based on whether its linear(L) or non-linear(Q)
if type =='L'
    %Plot straight line if boundary is linear
    boundX=[min(x(:,2))-2, max(x(:,2))+2];
    boundY=(-1./w(3)).*(w(2).*boundX+w(1));
    plot(boundX,boundY);
    title('Contours of logistic-linear model for P(L=1|x) as a function of 2-dimensional x');
    
elseif type=='Q'
    x_grid=linspace(min(x(:,2))-2, max(x(:,2))+2);
    y_grid=linspace(min(x(:,3))-2, max(x(:,3))+2);
    
    score= get_boundary(x_grid,y_grid, w);
    contour (x_grid, y_grid, score,[0,0]);
end 

elseif n==3 
    plot3(x(2,label==0),x(3,label==0),x(4,label==0),'o');hold on 
    plot3(x(2,label==1),x(3,label==1),x(4,label==1),'+');
     title('Contours of logistic-quadratic model for P(L=1|x) as a function of 2-dimensional x'),
   
end 

legend('Class 0', 'Class 1');
xlabel('x_1');ylabel('x_2'); hold on ; 

end 
