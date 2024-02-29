clear all
close all
clc

% Add paths
cd('../'); 
Install; 
%Choose the number of matrix and different initialization to perform
rep=5; init=5;

%Choose the sizes of the matrices to test
s=[500;1000;1500;2000]; q=length(s);

%Fix the rank
r=32; 

%Parameters setting
param.maxit=30000; param.tol=1e-4; param.tolerr = 0; param.time=1000;

%ANMD acceleration parameters
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%3B momentum parameters
param.beta1=0.7; 

%A-Naive algorithm
param.alpha1=0.5;

%Allocate variables
time_and=zeros(q,1); time_naive=zeros(q,1); time_naive_acc=zeros(q,1); 
iter_ANMD=zeros(q,1); iter_naive=zeros(q,1); iter_naive_acc=zeros(q,1); 

for i=1:length(s)
    %size of matrix
    m=s(i); n=m;    
        for k=1:rep
            %Generate the matrix
            W1=randn(m,r); H1=randn(r,n);  X=max(0,W1*H1); 

            for j=1:init
            %Nuclear norm initialization
            Theta1=randn(m,n);
            [Theta2,nuc] = nmd_nuclear_bt(X, Theta1, 3); 
            [ua,sa,va] = svds(Theta2,r); 
            svalues = diag(sa);
            param.W0 = ua; 
            param.H0 = sa*va';
            param.Theta0=param.W0*param.H0;
            
            %Naive-NMD
            [T_naive,err_naive,it_naive,t_naive]=Naive_NMD(X,r,param);
            iter_naive(i)=iter_naive(i)+it_naive(end);
            time_naive(i)=time_naive(i)+t_naive(end);

            %A-Naive
            [T_naive_acc,err_naive_acc,it_naive_acc,t_naive_acc]=A_Naive_NMD(X,r,param);
            iter_naive_acc(i)=iter_naive_acc(i)+it_naive_acc(end);
            time_naive_acc(i)=time_naive_acc(i)+t_naive_acc(end);

            %Naive with Andersens' acceleration
            [T_acc_and,err_acc_and,it_acc_and,t_and_acc]=A_NMD(X,r,param);
            iter_ANMD(i)=iter_ANMD(i)+it_acc_and(end);
            time_and(i)=time_and(i)+t_and_acc(end);
            
            end
        end

end

%Compute the average time needed to converge
time_naive=time_naive/(rep*init);
time_naive_acc=time_naive_acc/(rep*init);
time_and=time_and/(rep*init);

%Compute the average number of iterations needed to converge
iter_naive=iter_naive/(rep*init);
iter_naive_acc=iter_naive_acc/(rep*init);
iter_ANMD=iter_ANMD/(rep*init);

%Show results on tables
Size=s;

% Time table
Naive=time_naive ;
A_Naive=time_naive_acc;
ANMD=time_and;
G=table( Size, Naive, A_Naive, ANMD);
disp(G)

% Iterations table
Naive=round(iter_naive);
A_Naive=round(iter_naive_acc);
ANMD=round(iter_ANMD);
G=table( Size, Naive, A_Naive, ANMD);
disp(G)
