function score= get_boundary(hGrid,vGrid,theta)
%Generates grid of scores that spans the full range of data (where 
% a score of 0 indicates decision boundary level)
z=zeros(length(hGrid),length(vGrid));

for i=1:length(hGrid)
    for j =1: length(vGrid)
        %Map a the Quadratic function
        x_bound=[1 hGrid(i) vGrid(j) hGrid(i)^2 hGrid(i)*vGrid(j) hGrid(j)*vGrid(i) vGrid(j)^2];
        % Calculate score
        z(i,j)=x_bound*theta;
    end 
end 
score=z';
end 

 