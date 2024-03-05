%% Generating the matrix
rows = 1000;
cols = 1000;

% Generate random integers (0 or 1) for each element in the matrix
X = randi([0, 1], rows, cols);

fprintf("rank X= %d\n",[rank(X)]);



%% Run the algorithms 
% set the params
param=struct();
param.maxit=1000;
param.display=1;
param.tol=1e-10;

r=256+64;

[theta1,err1,i1,time1]= naive(X,r,param);
[theta2,err2,i2,time2]=A_NMD(X,r,param);




%% Plots for comparison

figure;

% Get the smallest positive floating-point number
epsilon = 1e-10;

% Plot err1 with circles as markers
plot(1:numel(err1), max(err1, epsilon), 'r-', 'LineWidth', 2); hold on;

% Plot err2 with squares as markers
plot(1:numel(err2), max(err2, epsilon), 'b-.', 'LineWidth', 2);
% Set axis labels and legend
xlabel('Time', 'FontSize', 14, 'FontName', 'Times');
ylabel('Relative Error', 'FontSize', 14, 'FontName', 'Times');
legend({'Err Binary decomposition', 'Err NMD'}, 'FontSize', 12, 'FontName', 'Times');
% Set y-axis limits to include zero
ylim([min([err1(:); err2(:)]), max([err1(:); err2(:)])]);

% Set y-axis scale to logarithmic
set(gca, 'YScale', 'log');

% Adjust tick marks and grid lines
set(gca, 'FontSize', 10);
grid on;
grid minor;
