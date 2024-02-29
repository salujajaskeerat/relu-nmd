% Script showing a basic example on synthetic data in order to understand
% how the codes for ReLu-NMD work

clear all
close all
clc
% Add paths
Install; 
%Fix the size of the matrix X
m=1000;n=1000; 

%Choose the approximation rank
r=32;

%Generate the synthetic matrix
W1=randn(m,r); H1=randn(r,n);  
X=max(0,W1*H1); 

%Run the codes with the default choice for the parameters

%A-NMD
[T_ANMD,err_ANMD,it_ANM,t_ANMD]=A_NMD(X,r);
%Get A-NMD approximation 
app_A_NMD=max(0,T_ANMD);
%Final error 
final_error_ANMD=norm(X-app_A_NMD,'fro')/norm(X,'fro');

%3B-NMD
[T_3B,err_3B,it_3B,t_3B]=NMD_3B(X,r);
%Get 3B-NMD approximation
app_3B_NMD=max(0,T_3B);
%Final error
final_error_3B=norm(X-app_3B_NMD,'fro')/norm(X,'fro');

%Plot the error per iteration
figure
subplot(1,2,1); 
semilogy(err_ANMD,'r--','LineWidth',1.5); hold on
semilogy(err_3B,'b--','LineWidth',1.5); 
grid on; 
xlabel('Iterations')
legend('A-NMD','3B-NMD')

%Plot the error as function of time
subplot(1,2,2); 
semilogy(t_ANMD,err_ANMD,'r--','LineWidth',1.5); hold on
semilogy(t_3B,err_3B,'b--','LineWidth',1.5); 
grid on; 
xlabel('Time (s.)')
legend('A-NMD','3B-NMD')