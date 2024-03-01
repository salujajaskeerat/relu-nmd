% check if we generate some same matrix and write it as sum of positive and negative matrices.
% We decompose these matrices into rank (r/2) each . will check if we can get better approximation than the tsvd


% Generate random matrix X
clear all;
clc
clc


rng(2023)


m=400;
n=200;
X=randn(m,n);

fprintf('rank of x = %d\n',[rank(X)]);

% Parameters
r=128; %rank
fprintf('r set to %d\n',[r]);

param=struct();
%Parameters setting
param.maxit=30000000; param.tol=1.e-4; param.tolerr = 0; param.time=20;
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

fprintf('rankf of Xplus= %d and Xminus= %d ',[rank(Xplus),rank(Xminus)]);

% Nucleur norma initialization
param.Theta0=Xplus;


%Set the interval for cubic spline interpolation for the average error
c=100;
time=linspace(0,param.time,c);


rep=1;
for k=1:rep
    %Naive with Andersens' acceleration
    [T_ANMD,err_ANMD,it_ANMD,t_ANMD]=A_NMD(Xplus,r/2,param);
    err_ANMD_k(k)=err_ANMD(end);                     %save final error
    p_ANMD(:,k) = spline(t_ANMD,err_ANMD,time); %interpolate in the desired points (needed to compute the average)

end


%Find best solution
opt=min(err_ANMD_k);

%Compute the average error
mean_ANMD=sum(p_ANMD')/rep;
% mean_3B=sum(p_3B')/rep;
% mean_AEM=sum(p_AEM')/rep;

%Subtract best solution
p_ANMD_mean=mean_ANMD-opt; 
% p_3B_mean=mean_3B-opt; 
% p_AEM_mean=mean_AEM-opt;

%Plot the error vs number of iteration

figure;

% Plot the data
plot(err_ANMD, 'r-', 'LineWidth', 2); hold on;
% plot(err_3B(1:300), 'b-.', 'LineWidth', 2);
% plot(err_AEM, 'g--', 'LineWidth', 2);

% Set axis labels and legend
xlabel('Time', 'FontSize', 18, 'FontName', 'Times');
ylabel('Relative Error', 'FontSize', 18, 'FontName', 'Times');
legend({'A-NMD', '3B-NMD', 'A-EM'}, 'FontSize', 18, 'FontName', 'Times');

% Set title with specific details
title(sprintf('X_plus and X_minus (r=%d)', r), 'FontSize', 20, 'FontName', 'Times');

% Set log scale for the y-axis
set(gca, 'YScale', 'log');

% Adjust tick marks and grid lines
set(gca, 'FontSize', 14);
grid on;

% Add grid lines with minor ticks for better readability
grid minor;