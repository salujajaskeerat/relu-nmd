% Average error analysis. 
% We compare A-NMD, 3B-NMD, A-EM. We display the quantity
%
%     err(t)=  ||X-max(0,Theta(t))||_F / || X ||_F - e_{min},
%
% where e_{min} is the lowest error computed by any algorithm with any
% initialization. The relative error is computed as an average over a
% number rep of runs. In order to compute the average we used a cubic
% spline approximation of the error.
% All algorithms are initialized using nuclear norm approach

clear all
close all
clc

% Add paths
cd('../../'); 
Install; 

%% Load Dataset and define rank

%Choose the approximation rank
r=32; 

%Choose between synthetic data or MNIST dataset
number=1;
switch number
    case 1 %noiseless synthetic data
        m=200; n=200; %dimension of the problem
        W1=randn(m,r); H1=randn(r,n);  X=max(0,W1*H1); 
    case 2 %MNIST dataset
        Y=load('mnist_all.mat');
        w1=1:10:500; %Number of images for each digit
        %w1=1:5000;
        X=[Y.train0(w1,:);Y.train1(w1,:);Y.train2(w1,:);Y.train3(w1,:);Y.train4(w1,:);...
           Y.train5(w1,:);Y.train6(w1,:);Y.train7(w1,:);Y.train8(w1,:);Y.train9(w1,:)];
        X=double(X);
        [m,n]=size(X);
end


%% Set the parameters
%Parameters setting
param_nesterov = struct('maxit', 300000, 'tol', 1e-4, 'tolerr', 0, 'time', 10, ...
    'beta', 0.7, 'eta', 0.4, 'gamma', 1.1, 'gamma_bar', 1.05, ...
    'beta1', 0.7, 'alpha', 0.6);

param_anderson=struct('depth',3,'freeze',6,'maxit',1000,'time',20,'tol',1e-4,'tolerr',1e-6,'C',10);



%% Run all baseline and anderson algorithm

%Set the interval for cubic spline interpolation for the average error
c=100;
time=linspace(0,param_nesterov.time,c);


rep=1;
for k=1:rep

    %Random initialization
    %alpha=sum(sum(X.*Z0))/norm(Z0,'fro')^2;
    %param_nesterov.W0=alpha*randn(n,r); param_nesterov.H0=(randn(r,m));
    %param_nesterov.Theta=param_nesterov.W0*param_nesterov.H0
    %Nuclear norm initialization
%     Theta1=randn(m,n);
%     [Theta2,nuc] = nmd_nuclear_bt(X, Theta1, 3); 
%     [ua,sa,va] = svds(Theta2,r); 
%     svalues = diag(sa);
%     param_nesterov.W0 = ua; 
%     param_nesterov.H0 = sa*va';
%     param_nesterov.Theta0=param_nesterov.W0*param_nesterov.H0;
%     
    %Naive with Andersens' acceleration
    [T_ANMD,err_ANMD,it_ANMD,t_ANMD]=A_NMD(X,r,param_nesterov);
    err_ANMD_k(k)=err_ANMD(end);                     %save final error
    p_ANMD(:,k) = spline(t_ANMD,err_ANMD,time); %interpolate in the desired points (needed to compute the average)

%     %Accelerated three blocks algorithm
%     [T_3B,err_3B,it_3B,t_3B]=NMD_3B(X,r,param_nesterov);
%     err_3B_k(k)=err_3B(end);
%     p_3B(:,k)=spline(t_3B,err_3B,time);
% 

%     %Expectation-minimization by Saul
%     [T_AEM,err_AEM,it_AEM,t_AEM] = A_EM_NMD(X,r,param_nesterov);
%     err_AEM_k(k)=err_AEM(end);
%     p_AEM(:,k)=spline(t_AEM,err_AEM,time);

    % Expectation maximisation using anderson acceleration
    [T_ANDR,err_ANDR,i_ANDR,t_ANDR]=anderson(X,r,param_anderson);

end



%% Varying depth 
depth_results = struct();
[T_ANMD,err_ANMD,it_ANMD,t_ANMD]=A_NMD(X,r,param_nesterov);
for d = 2:5
    param_anderson.depth = d;
    [T_ANDR, err_ANDR, i_ANDR, t_ANDR,Z,alphas] = anderson(X, r, param_anderson);
    
    % Store the results in the depth_results structure
    depth_results.(['depth_', num2str(d)]) = struct('T_ANDR', T_ANDR, 'err_ANDR', err_ANDR, 'i_ANDR', i_ANDR, 't_ANDR', t_ANDR,'alphas',alphas);
end

% Display the depth_results structure
disp(depth_results);

%% Plot depth vs error


figure;
hold on;

% Iterate over the fields of depth_results
fields = fieldnames(depth_results);
plot( log10(err_ANMD), 'LineWidth', 2);
for i = 1:numel(fields)
    % Extract the results for the current depth
    depth_data = depth_results.(fields{i});
    
    % Plot err_ANDR vs. iteration number for the current depth
    plot( log10(depth_data.err_ANDR), 'LineWidth', 2);
end

% Add legend with appropriate labels for each depth
legend_str = ['ANMD'; cellfun(@(x) ['Depth ' num2str(str2double(strrep(x, 'depth_', '')))], fields, 'UniformOutput', false)];
legend(legend_str, 'Location', 'Best');

% Set labels and title
xlabel('Iteration Number');
ylabel('Error');
title('Error vs. Iteration Number for Different Depths');
% Set y-axis scale to logarithmic
% set(gca, 'YScale', 'log');
grid on;
grid minor;

hold off;


