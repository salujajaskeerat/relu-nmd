% check if we generate some same matrix and write it as sum of positive and negative matrices.
% We decompose these matrices into rank (r/2) each . will check if we can get better approximation than the tsvd


% Generate random matrix X
clear all;
clc
clc


rng(2023)


m=1000;
n=1000;
X=randi([-1000,1000],m,n);

fprintf('rank of x = %d\n',[rank(X)]);

% Parameters
%r=512; %rank


param=struct();
%Parameters setting
param.maxit=1000; param.tol=1.e-4; param.tolerr = 0; param.time=20;
%A-NMD parameters
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;
%3 Blocks momentum parameters
param.beta1=0.7; 
% A-EM parameter
param.alpha=0.6;


% results with TSVD


% results with X=X^+ - X^-
Xplus= max(0,X);
Xminus=max(0,-X);

fprintf('rankf of Xplus= %d and Xminus= %d\n ',[rank(Xplus),rank(Xminus)]);

% Nucleur norma initialization
% param.Theta0=Xplus;


%Set the interval for cubic spline interpolation for the average error
c=100;
time=linspace(0,param.time,c);


normX=norm(X,'fro');


log_rank=  int64(floor(log2(rank(X))));
for i=log_rank-2:log_rank
    r=2^i;
    fprintf('r set to %d\n',[r]);
    rep=1;
    for k=1:rep
        [theta_XPLUS,err_XPLUS,i_XPLUS,time_XPLUS]=A_NMD(Xplus,r,param);
        % for Xminus
        [theta_XMINUS,err_XMINUS,i_XMINUS,time_XMINUS]=A_NMD(Xminus,r,param);
    
        % Compute err for original X
        err_X= norm((X-max(0,theta_XPLUS)+max(0,theta_XMINUS)),'fro')/normX;
    end
    %Plot the error vs number of iteration
    figure;
    % Plot the data
    x_range=linspace(1,max(size(err_XMINUS,2),size(err_XPLUS,2)),100);
    plot(err_XPLUS, 'r-', 'LineWidth', 2); hold on;
    plot(err_XMINUS, 'b-.', 'LineWidth', 2);
    plot(x_range,err_X*ones(size(x_range)),'cyan-.','LineWidth',1.5);
    plot(x_range,(min(err_XPLUS)+min(err_XMINUS))*ones(size(x_range)),'black--','LineWidth',1.5);
    % Set axis labels and legend
    xlabel('Time', 'FontSize', 18, 'FontName', 'Times');
    ylabel('Relative Error', 'FontSize', 18, 'FontName', 'Times');
    legend({'Err XPLUS', 'Err XMINUS','ERR X','final Err XPLUS + final Err XMINUS'}, 'FontSize', 18, 'FontName', 'Times');
    % Set title with specific details
    title(sprintf('Err XPLUS and XMINUS (r=%d)', r));
    % Set log scale for the y-axis
    set(gca, 'YScale', 'log');
    % Adjust tick marks and grid lines
    set(gca, 'FontSize', 14);
    grid on;
    % Add grid lines with minor ticks for better readability
    grid minor;
end