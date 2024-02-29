%In the Script we compute the average time and iterations for rep different matrices, for a
%number init of different initializations, post-processed using nuclear norm
%algorithm.
%We repeat the experiments for different values of the rank r, collected on 
%the vector ra and we display the final results on a table.
%We test A-NMD, 3B-NMD, EM-NMD, A-EM.


clear all
close all
clc

% Add paths
cd('../'); 
Install; 

%Choose the number of matrix and different initialization to perform
rep=5; init=5;

%Fix the size of the matrix X
m=1000;n=1000; 

%Choose the approximation rank to test 
ra=[8;16;32;64];  q=length(ra);

%Parameters setting
param.maxit=30000; param.tol=1e-4; param.tolerr = 0; param.time=1000;

%Andersen's acceleration parameters
param.beta=0.7; param.eta=0.4; param.gamma=1.1; param.gamma_bar=1.05;

%3 Blocks momentum parameters
param.beta1=0.7; 

%Allocate variables
time_ANMD=zeros(q,1); time_3B=zeros(q,1); time_EM=zeros(q,1); time_AEM=zeros(q,1);
iter_ANMD=zeros(q,1); iter_3B=zeros(q,1); iter_EM=zeros(q,1); iter_AEM=zeros(q,1);

for i=1:length(ra)
    %Approximation rank
    r=ra(i);

        for k=1:rep
            %Generate the matrix
            W1=randn(m,r); H1=randn(r,n);  X=max(0,W1*H1); 

            for j=1:init
            %Nuclear norm initialization initialization
            Theta1=randn(m,n);
            [Theta2,nuc] = nmd_nuclear_bt(X, Theta1, 3); 
            [ua,sa,va] = svds(Theta2,r);  
            param.W0 = ua; 
            param.H0 = sa*va';
            param.Theta0=param.W0*param.H0;
                    
            %A-NMD 
            [T_ANMD,err_ANMD,it_ANMD,t_ANMD]=A_NMD(X,r,param);
            iter_ANMD(i)=iter_ANMD(i)+it_ANMD; 
            time_ANMD(i)=time_ANMD(i)+t_ANMD(end);
            
            %3B-NMD 
            [T_3B,err_3B,it_3B,t_3B]=NMD_3B(X,r,param);
            iter_3B(i)=iter_3B(i)+it_3B;
            time_3B(i)=time_3B(i)+t_3B(end);
        
            %EM-NMD by Saul
            [T_EM,err_EM,it_EM,t_EM] = EM_NMD(X,r,param);
            iter_EM(i)=iter_EM(i)+it_EM;
            time_EM(i)=time_EM(i)+t_EM(end);

            %A-EM by Saul
            [T_AEM,err_AEM,it_AEM,t_AEM] = A_EM_NMD(X,r,param);
            iter_AEM(i)=iter_AEM(i)+it_AEM;
            time_AEM(i)=time_AEM(i)+t_AEM(end);
            end
        end

end

%Compute the average time needed to converge
time_ANMD=time_ANMD/(rep*init);
time_3B=time_3B/(rep*init);
time_EM=time_EM/(rep*init);
time_AEM=time_AEM/(rep*init);

%Compute the average number of iterations needed to converge
iter_ANMD=iter_ANMD/(rep*init);
iter_3B=iter_3B/(rep*init);
iter_EM=iter_EM/(rep*init);
iter_AEM=iter_AEM/(rep*init);

%Show results on tables
Rank=ra;
ANMD=time_ANMD;
B3_NMD=time_3B ;
EM_NMD=time_EM;
EM_NMD_mom=time_AEM;
G=table( Rank, ANMD, B3_NMD, EM_NMD,EM_NMD_mom);
disp(G)

ANMD=round(iter_ANMD);
B3_NMD=round(iter_3B) ;
EM_NMD=round(iter_EM);
EM_NMD_mom=round(iter_AEM);
G=table( Rank, ANMD, B3_NMD, EM_NMD,EM_NMD_mom);
disp(G)

