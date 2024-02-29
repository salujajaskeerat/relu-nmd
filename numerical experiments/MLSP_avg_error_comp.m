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
cd('../'); 
Install; 

rep=2;  %number of runs for each algorithm---> it must be r>1

%Choose the approximation rank
r=32; 

%Choose between synthetic data or MNIST dataset
number=2;
switch number
    case 1 %noiseless synthetic data
        m=4000; n=2000; %dimension of the problem
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


%Parameters setting
param.maxit=300000; param.tol=1.e-4; param.tolerr = 0; param.time=10;

%A-NMD acceleration parameters
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%3B-NMD momentum parameters
param.beta1=0.7; 

%A-EM momentum parameter
param.alpha=0.6;

%Set the interval for cubic spline interpolation for the average error
c=100;
time=linspace(0,param.time,c);

for k=1:rep

    %Random initialization
    %alpha=sum(sum(X.*Z0))/norm(Z0,'fro')^2;
    %param.W0=alpha*randn(n,r); param.H0=(randn(r,m));
    %param.Theta=param.W0*param.H0
    %Nuclear norm initialization
    Theta1=randn(m,n);
    [Theta2,nuc] = nmd_nuclear_bt(X, Theta1, 3); 
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
