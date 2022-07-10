clear all, close all,

n = 2; N = 1000;
wtrue = 6*randn(n,1),
A = randn(n,n); x = A*randn(n,N);
y = wtrue'*x + 1e-2*randn(1,N);

hGrid = linspace(-20,20,101);
vGrid = linspace(-20,20,99);
[h,v] = meshgrid(hGrid,vGrid);
for HG = 1:length(hGrid)
    for VG = 1:length(vGrid)
        mseGrid(VG,HG) = costFunctionAverageSquaredError(x,y,[hGrid(HG);vGrid(VG)]);
    end
end
figure(1), subplot(2,1,1),
contour(hGrid,vGrid,mseGrid); axis equal, hold on, 

w(:,1) = wtrue + 3*(randn(n,1)+5*rand(n,1)); 
alpha = 1e-1;
T = 2000; ind = zeros(1,T); e = zeros(1,T);
for k = 1:T
    ind(1,k) = randi([1,N],1); % pick a sample randomly from the training set
    e(1,k) = (y(ind(k))-w(:,k)'*x(:,ind(k)));
    figure(1), subplot(2,1,2), semilogy(k,e(1,k)^2,'.'), hold on, grid on, 
    xlabel('Iterations'), ylabel('Instantaneous Squared Error'),
    title('With a suitable step size, you should see a downward trend...'),
    drawnow,
    figure(1), subplot(2,1,1), plot(w(1,k),w(2,k),'.'), axis equal, hold on,
    plot(wtrue(1),wtrue(2),'+r'),axis equal,
    xlabel('w_1'), ylabel('w_2'),
    title('True Weight Vector (+) and Estimated Weights At Each Iteration (.)'),
    drawnow,
    w(:,k+1) = w(:,k) + 2*alpha*(y(ind(k))-w(:,k)'*x(:,ind(k)))*x(:,ind(k));
end

