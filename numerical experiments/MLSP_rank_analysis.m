% Rank analysis
% In the script we compare TSVD, A-NMD, 3B-NMD, and A-EM on MNIST dataset evaluating the
% relative error and the computational time for increasing values of the approximation rank.
% 
% We plot two graphs displaying the relative error and the time per
% iteration for increasing values of the rank.

clear all
close all
clc

% Add paths
cd('../'); 
Install; 

rng(2023)

%Parameters setting
param.maxit=30000000; param.tol=1.e-4; param.tolerr = 0; param.time=20;

%A-NMD parameters
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%3 Blocks momentum parameters
param.beta1=0.7; 

% A-EM parameter
param.alpha=0.6;

%load MNIST dataset
Y=load('mnist_all.mat');
w1=1:5000;    %choose how many images per digit to include in matrix X
X=[Y.train0(w1,:);Y.train1(w1,:);Y.train2(w1,:);Y.train3(w1,:);Y.train4(w1,:);...
   Y.train5(w1,:);Y.train6(w1,:);Y.train7(w1,:);Y.train8(w1,:);Y.train9(w1,:)];
X=double(X);
[m,n]=size(X);

vec=[8,16,32,64,128,256]; %Choose the values of the rank to test

for j=1:length(vec)
    r=vec(j); %approximation rank
    fprintf('\n Running experimnt for rank=%d \n',r);

    %Random initialization
    % alpha=sum(sum(X.*Z0))/norm(Z0,'fro')^2;
    % param.W0=alpha*randn(n,r); param.H0=(randn(r,m));
    % param.Theta0=param.W0*param.H0;

    %Nuclear norm initializing strategy
    normX=norm(X,'fro');
    Theta1=randn(m,n);
    [Theta2,nuc] = nmd_nuclear_bt(X, Theta1, 3); 
    [ua,sa,va] = svds(Theta2,r); 
    svalues = diag(sa);
    param.W0 = ua; 
    param.H0 = sa*va';
    param.Theta0=param.W0*param.H0;
    
    %SVD computation for comparison
    [U,S,V]=svds(X,r);
    SVD_ap=U*S*V';
    err_svd(j)=norm(X-max(0,SVD_ap),'fro')/normX;
    
    %A-NMD
    [T_ANMD,err_ANMD,it_ANMD,t_ANMD]=A_NMD(X,r,param);
    time_ANMD(j)=t_ANMD(end)/it_ANMD;                    %time per iteration
    err_ANMD_final(j)=norm(X-max(0,T_ANMD),'fro')/normX; %final relative error
    
    %3B-NMD
    [T_3B,err_3B,it_3B,t_3B]=NMD_3B(X,r,param);
    time_3B(j)=t_3B(end)/it_3B;
    err_3B_final(j)=norm(X-max(0,T_3B),'fro')/normX;
    
    
    %A-EM
    [T_AEM,err_AEM,it_AEM,t_AEM] = A_EM_NMD(X,r,param);
    time_AEM(j)=t_AEM(end)/it_AEM;
    err_AEM_final(j)=norm(X-max(0,T_AEM),'fro')/normX;


end

%Display relative error graph
figure
y=[0,linspace(10,vec(end),length(vec))]; 
plot(y(2:end),err_svd,'-md','LineWidth',1.5); hold on
plot(y(2:end),err_ANMD_final,'-rs','LineWidth',1.5); hold on
plot(y(2:end),err_3B_final,'-b^','LineWidth',1.5);
plot(y(2:end),err_AEM_final,'-ko','LineWidth',1.5);
legend({'TSVD','A-NMD','3B-NMD','A-EM'},'FontSize',22,'FontName','times')
xlabel('Rank','FontSize',22,'FontName','times'); ylabel('Relative error','FontSize',22,'FontName','times')
xticks(y);
xticklabels({'','8','16','32','64','128','256'})   %change labels if change vec
grid on

%Display average iteration time 
figure
y=[0,linspace(10,vec(end),length(vec))]; 
plot(y(2:end),time_ANMD,'-rs','LineWidth',1.5); hold on
plot(y(2:end),time_3B,'-b^','LineWidth',1.5);
plot(y(2:end),time_AEM,'-ko','LineWidth',1.5);
legend({'A-NMD','3B-NMD','A-EM'},'FontSize',22,'FontName','times')
xlabel('Rank','FontSize',22,'FontName','times'); ylabel('Second per iteration','FontSize',22,'FontName','times')
xticks(y);
xticklabels({'','8','16','32','64','128','256'})    %change labels if change vec
grid on
