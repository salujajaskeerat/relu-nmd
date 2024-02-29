% In this script we use ReLU-NMD algorithms in order to compress the sparse
% factor of the NMF factorization. At first we use minvol_NMF to obtain U,V
% such that X is approximated by UV. Note that U is tipically sparse.
% We then use NNLS to compute the best possible V_best and we evaluate the NMF
% error as
%           
%         || X - U*V_best ||_F / || X ||_F.
%
% Then we apply ReLU-NMD algorithms to further approximate U, we obtain a 
% low-rank approximation U1 and we evaluate the results computing the NMD
% error
%
%         || U - max(0,U1) ||_F / || U  ||_F.
%
% Finally we plot the results for incresing values of the rank
% approximation of the NMD.

disp('In order to run this code, you need the MATLAB NMF toolbox from: https://gitlab.com/ngillis/nmfbook/'); 

clear all
close all
clc

% Add paths
cd('../'); 
Install; 

%Parameters setting
param.maxit=50000000; param.tol=1.e-4; param.tolerr = 0; param.time=20;

%A-NMD acceleration parameters
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%3B-NMD momentum parameters
param.beta1=0.7; 

%A-EM momentum parameter
param.alpha=0.6;

%Load the CBCL 19x19 faces data set
load CBCL 
s = 100;      %approximation rank of the NMF factors
%Compute NMF using minvol-NMF
[V,U] = minvolNMF(X',s); 
U = U'; [m,n] = size(U);

%plot the original U factor
U_orig=affichage(U,10,19,19); 

%Compute the best V
V = NNLS(U,X); 

%NMF error evaluation 
err_NMF_real=norm(X-U*V,'fro')/norm(X,'fro');

%Choose the approximation ranks
vec=[8,12,16,20,24];

for j=1:length(vec)
    r=vec(j); %approximation rank
    fprintf('\n Running experimnt for rank=%d \n',r);

    %Random initialization
    % alpha=sum(sum(X.*Z0))/norm(Z0,'fro')^2;
    % W0=alpha*randn(n,r); H0=(randn(r,m));
    % Theta0=W0*H0;
    %initializing strategy
    Theta1=randn(m,n);
    [Theta2,nuc] = nmd_nuclear_bt(U, Theta1, 3); 
    [ua,sa,va] = svds(Theta2,r); 
    svalues = diag(sa);
    param.W0 = ua; 
    param.H0 = sa*va';
    param.Theta0=param.W0*param.H0;
    
    %SVD computation for comparison
    [U1,S,V1]=svds(U,r);
    U_SVD_neg=U1*S*V1';
    U_SVD=max(0,U_SVD_neg);
    err_SVD(j)=norm(U-U_SVD,'fro')/norm(U,'fro');
    V_SVD = NNLS(U_SVD,X);
    err_SVD_nmf(j)=norm(X-U_SVD*V_SVD,'fro')/norm(X,'fro');
    
    %Naive with Andersens' acceleration
    [U_ANMD_neg,err_acc_and,it_acc_and,t_and_acc]=A_NMD(U,r,param);
    U_ANMD=max(0,U_ANMD_neg);
    time_ANMD(j)=t_and_acc(end)/it_acc_and;
    err_ANMD(j)=norm(U-U_ANMD,'fro')/norm(U,'fro');            %NMD error
    V_ANMD = NNLS(U_ANMD,X);                                   %Best V
    err_ANMD_nmf(j)=norm(X-U_ANMD*V_ANMD,'fro')/norm(X,'fro'); %NMF error
    
    %Accelerated three blocks algorithm
    [U_3B_neg,err_nmf_mom,it_3B,t_3B]=NMD_3B(U,r,param);
    U_3B=max(0,U_3B_neg);
    time_3B(j)=t_3B(end)/it_3B;
    err_3B(j)=norm(U-U_3B,'fro')/norm(U,'fro');
    V_3B = NNLS(U_3B,X); 
    err_3B_nmf(j)=norm(X-U_3B*V_3B,'fro')/norm(X,'fro');
    
    %Expectation-minimization by Saul
    [U_EM_neg,err_em,it_saul,t_saul] = A_EM_NMD(U,r,param);
    U_EM=max(0,U_EM_neg);
    time_EM(j)=t_saul(end)/it_saul;
    err_EM(j)=norm(U-U_EM,'fro')/norm(U,'fro');
    V_EM = NNLS(U_EM,X); 
    err_EM_nmf(j)=norm(X-U_EM*V_EM,'fro')/norm(X,'fro');
end

%Plot NMD error comparison
figure
y=[0,linspace(5,vec(end),length(vec))]; 
semilogy(y(2:end),err_SVD,'-md','LineWidth',1.5); hold on
semilogy(y(2:end),err_ANMD,'-rs','LineWidth',1.5); hold on
semilogy(y(2:end),err_3B,'-b^','LineWidth',1.5);
semilogy(y(2:end),err_EM,'-ko','LineWidth',1.5);
legend({'TSVD','A-NMD','3B-NMD','A-EM'},'FontSize',22,'FontName','times')
xlabel('Rank','FontSize',22,'FontName','times'); ylabel('NMD relative error','FontSize',22,'FontName','times')
xticks(y);
xticklabels({'0','8','12','16','20','24'})    %if change vec change also theese labels
grid on

%Plot NMF error comparison
figure
y=[0,linspace(5,vec(end),length(vec))]; 
semilogy(y(2:end),err_SVD_nmf,'-md','LineWidth',1.5); hold on
semilogy(y(2:end),err_ANMD_nmf,'-rs','LineWidth',1.5); 
semilogy(y(2:end),err_3B_nmf,'-b^','LineWidth',1.5);
semilogy(y(2:end),err_EM_nmf,'-ko','LineWidth',1.5);
semilogy(y(2:end),ones(length(vec),1)*err_NMF_real,'-g','LineWidth',1.5);
legend({'TSVD','A-NMD','3B-NMD','A-EM','NMF error'},'FontSize',18,'FontName','times')
xlabel('Rank','FontSize',22,'FontName','times'); ylabel('NMF relative error','FontSize',22,'FontName','times')
xticks(y);
xticklabels({'0','8','12','16','20','24'})     %if change vec change also theese labels
grid on

