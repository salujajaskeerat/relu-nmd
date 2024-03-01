function [Theta,err,i,time]=A_NMD(X,r,param)

% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{Z,Theta} ||Z-Theta||_F^2  s.t. rank(Theta)=r, max(0,Z)=X
%
% using an alternating procedure + adaptive extrapolation technique;
%
%****** Input ******
%   X       : m-by-n matrix, sparse and non negative
%   Theta0  : m-by-n matrix
%   r       : scalar, desired approximation rank
%   param   : structure, containing the parameters of the algorithm
%       .Theta0 = initialization of the variable Theta (default: randn)
%       .maxit  = maximum number of iterations (default: 1000)
%       .tol    = tolerance on the relative error (default: 1.e-4)
%       .tolerr = tolerance on 10 successive errors (err(i+1)-err(i-10)) (default: 1.e-5)
%       .time   = time limit (default: 20)
%       .eta,gamma,gamma_bar = hyperparameters such thateta<1<beta_bar<beta (default: 0.4<1<1.05<1.1)
%       .display= if set to 1, it diplays error along iterations (default: 1)
% ****** Output ******
%   Theta   : m-by-n matrix, approximate solution of
%             min_{Theta}||X-max(0,Theta)||_F^2  s.t. rank(Theta)=r.
%   err     : vector containing evolution of relative error along
%             iterations ||X-max(0,Theta)||_F / || X ||_F
%   i       : number of iterations
%   time    : vector containing time counter along iterations
% 
% See the paper ''Accelerated Algorithms for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Atharva Awari, Arnaud 
% Vandaele, Margherita Porcelli, and Nicolas Gillis, 2023. 


[m, n] = size(X);
defaults = struct('Theta0', randn(m, n), 'maxit', 1000, 'tol', 1e-4, 'tolerr', 1e-5, 'time', 20, 'beta', 0.9, 'eta', 0.4, 'gamma', 1.1, 'gamma_bar', 1.05, 'display', 1);
if nargin < 3
    param = defaults;
else
    param = setfield(defaults, param);
end


%Inizialization of parameters of the model
beta_bar=1;
beta_history(1)=param.beta;

%Detect (negative and) positive entries of X
if min(X(:)) < 0
    warnmess1 = 'The input matrix should be nonnegative. \n';
    warnmess2 = '         The negative entries have been set to zero.';
    warning(sprintf([warnmess1 warnmess2]));
    X(X<0) = 0;
end
[m,n]=size(X);
normX=norm(X,'fro');
idx=(X==0);
idxp=(X>0);

%Initialize the latent variable
Z0 = zeros(m,n);
Z0(idxp) = nonzeros(X);

%Create istances for variables
Z=Z0; Theta=param.Theta0; Z_old=Z0; Theta_old=param.Theta0;

%Initialize error and time counter
err(1)=norm(max(0,Theta)-X,'fro')/normX;
time(1)=0;

if param.display == 1
    disp('Running A-NMD, evolution of [iteration number : relative error in %]');
end

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.1;
for i=1:param.maxit
    tic
    %Update on Z ---> Z=min(0,Theta)
    Z=min(0,Theta.*idx);
    Z=Z+X.*idxp;
    
    %Momentum on Z
    Z=Z+param.beta*(Z-Z_old);
    
    %Update of T
    [W,D,V] = tsvd(Z,r);  %function computing TSVD
    H = D*V'; % H=DV matrix multiplication
    Theta = W*H;
    Theta_return=Theta;
    
    %Error computation
    Ap=max(0,Theta);
    err(i+1)=norm(Ap-X,'fro')/normX; 
    %Standard stopping condition on the relative error
    if err(i+1)<param.tol
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
        if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: ||X-max(0,WH)||/||X|| < %2.0d\n',param.tol);
        end
        break
    end
    if i >= 11  &&  abs(err(i+1) - err(i-10)) < param.tolerr
        time(i+1)=time(i)+toc; %needed to have same time components as iterations
        if param.display == 1
            if mod(numdis,5) > 0, fprintf('\n'); end
            fprintf('The algorithm has converged: rel. err.(i+1) - rel. err.(i+10) < %2.0d\n',param.tolerr);
        end
        break
        break
    end
    
    %Momentum step on Theta, not in the last iteration otherwise Theta has not rank r
    if i<param.maxit-1
        Theta=Theta+param.beta*(Theta-Theta_old);
    end
    
    %Quantity to check for parameter update
    back(i)=norm(max(0,Theta)-X,'fro')/normX;
    if i>2
        if back(i)<back(i-1)
            param.beta=min(beta_bar,param.gamma*param.beta);  %Update momentum parameter
            beta_bar=min(1,param.gamma_bar*param.beta);       %Upper bound update
            beta_history(i)=param.beta;                       %Keep trace of parameters
            
            %Accept the update of Z
            Z_old=Z; Theta_old=Theta;
        else
            param.beta=param.eta*param.beta;      %Update momentum parameter
            beta_history(i)=param.beta;           %Keep trace of parameters
            beta_bar=beta_history(i-2);           %Upper bound update: last value that allowed
            %the decrease of objective function
            
            %Do not accept the update of Z
            Z=Z_old; Theta=Theta_old;
        end
    end
    
    %Stopping condition on time
    time(i+1)=time(i)+toc;
    if time(i+1)>param.time
        Theta=Theta_return;
        break
    end
    
    if param.display == 1 && time(i+1) >= cntdis
        disp_time = min(60,disp_time*1.5); 
        fprintf('[%2.2d : %2.2f] - ',i,100*err(i+1));
        cntdis = time(i+1)+disp_time; % display every disp_time
        numdis = numdis+1;
        if mod(numdis,5) == 0
            fprintf('\n');
        end
    end
    
end
if param.display == 1
    fprintf('Final relative error: %2.2f%%, after %2.2d iterations. \n',100*err(i+1),i);
end