% Rank analysis for the ciphar dataset. 

clear all
clear all
clc

% Add path
cd('../');
Install

% Load training data
data = load('cifar-10-batches-mat/data_batch_1.mat');
rng(2023)

%Parameters setting
param.maxit=30000000; param.tol=1.e-4; param.tolerr = 0; param.time=20;

%A-NMD parameters
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%3 Blocks momentum parameters
param.beta1=0.7; 

% A-EM parameter
param.alpha=0.6;



X=data.data(1:1000,:);
X=double(X);
[m,n]=size(X);

%Set the interval for cubic spline interpolation for the average error
c=100;
time=linspace(0,param.time,c);

% ranks
r=32;

rep=1;
for k=1:rep

    %Random initialization
    %alpha=sum(sum(X.*Z0))/norm(Z0,'fro')^2;
    %param.W0=alpha*randn(n,r); param.H0=(randn(r,m));
    %param.Theta=param.W0*param.H0
    %Nuclear norm initialization
    
    % randomly assign a matrix m*n 
    Theta1=randn(m,n);


    % use the nuclear norm on this random initialization and X
    [Theta2,nuc] = nmd_nuclear_bt(X, Theta1, 3); 

    % decomposr the matrix Theta2=W*H
    [ua,sa,va] = svds(Theta2,r); 
    svalues = diag(sa);
    param.W0 = ua; 
    param.H0 = sa*va';
    param.Theta0=param.W0*param.H0;
    
    %Naive with Andersens' acceleration
    [T_ANMD,err_ANMD,it_ANMD,t_ANMD]=A_NMD(X,r,param);
    err_ANMD_k(k)=err_ANMD(end);                     %save final error
    p_ANMD(:,k) = spline(t_ANMD,err_ANMD,time); %interpolate in the desired points (needed to compute the average)

    %Accelerated three blocks algorithm
    [T_3B,err_3B,it_3B,t_3B]=NMD_3B(X,r,param);
    err_3B_k(k)=err_3B(end);
    p_3B(:,k)=spline(t_3B,err_3B,time);

    %Expectation-minimization by Saul
    [T_AEM,err_AEM,it_AEM,t_AEM] = A_EM_NMD(X,r,param);
    err_AEM_k(k)=err_AEM(end);
    p_AEM(:,k)=spline(t_AEM,err_AEM,time);

end




%Find best solution
opt=min([err_ANMD_k,err_3B_k,err_AEM_k]);

%Compute the average error
mean_ANMD=sum(p_ANMD')/rep;
mean_3B=sum(p_3B')/rep;
mean_AEM=sum(p_AEM')/rep;

%Subtract best solution
p_ANMD_mean=mean_ANMD-opt; 
p_3B_mean=mean_3B-opt; 
p_AEM_mean=mean_AEM-opt;

%Plot the results
figure
set(gca,'Fontsize',18)
semilogy(time,p_ANMD_mean,'r--','LineWidth',1.5); hold on
semilogy(time,p_3B_mean,'b-.','LineWidth',1.9); 
semilogy(time,p_AEM_mean,'k-','LineWidth',1.9);
xlabel('Time','FontSize',22,'FontName','times'); ylabel('err(t)','FontSize',22,'FontName','times');
legend({'A-NMD','3B-NMD', 'A-EM'},'FontSize',22,'FontName','times')


%Plot the error vs number of iteration

figure
plot(err_ANMD,'r--','LineWidth',1/5);hold on 
plot(err_3B,'b--','LineWidth',1/5); 
plot(err_AEM,'g--','LineWidth',1/5); 

xlabel('Time','FontSize',22,'FontName','times'); ylabel('relative error','FontSize',22,'FontName','times');
legend({'A-NMD','3B-NMD', 'A-EM'},'FontSize',22,'FontName','times')
title(sprintf('r=%d , Cifar dataset',r),'FontSize',22,'FontName','times')





% Figure 