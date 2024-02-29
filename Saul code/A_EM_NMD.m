function [Theta,err,i,time] = A_EM_NMD(X,r,param)

% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{Z,Theta} ||Z-Theta||_F^2  s.t. rank(Theta)=r, max(0,Z)=X 
% 
% using an expectation-minimization approach and a momentum step with fixed parameter
%        
%       Z^{k+1}=Z^{k+1}+alpha*(Z^k-Z^{k-1})
% 
%****** Input ******
%   X       : m-by-n matrix, sparse and non negative
%   Theta0  : m-by-n matrix
%   r       : scalar, desired approximation rank
%   param   : structure, containing the parameter of the model
%       .Theta0 = initialization of the variable Theta (default: randn)
%       .maxit  = maximum number of iterations (default: 1000) 
%       .tol    = tolerance on the relative error (default: 1.e-4)
%       .tolerr = tolerance on 10 successive errors (err(i+1)-err(i-10)) (default: 1.e-5)
%       .time   = time limit (default: 20)
%       .alpha  = momentum parameter (default: 0.6)
%       .display= if set to 1, it diplayes error along iterations (default: 1)
% 
% ****** Output ******
%   Theta   : m-by-n matrix, approximate solution of    
%             min_{Theta}||X-max(0,Theta)||_F^2  s.t. rank(Theta)=r.
%   err     : vector containing evolution of relative error along
%             iterations ||X-max(0,Theta)||_F / || X ||_F
%   i       : number of iterations  
%   time    : vector containing time counter along iterations
%
% Saul, L. K. (2022). A nonlinear matrix decomposition for mining the zeros of sparse data. 
% SIAM Journal on Mathematics of Data Science, 4(2), 431-463.

[m,n]=size(X); 
 if nargin < 3
    param = [];
 end
 if ~isfield(param,'Theta0') 
     param.Theta0=randn(m,n);  
 end
if ~isfield(param,'maxit')
    param.maxit = 1000; 
end
if ~isfield(param,'tol')
    param.tol = 1.e-4; 
end
if ~isfield(param,'tolerr')
    param.tolerr = 1e-5;
end
if ~isfield(param,'time')
    param.time = 20; 
end
if ~isfield(param,'alpha')
    param.alpha = 0.6; 
end
if ~isfield(param,'display')
    param.display = 1;
end

%Detect (negative and) positive entries of X
if min(X(:)) < 0
    warnmess1 = 'The input matrix should be nonnegative. \n'; 
    warnmess2 = '         The negative entries have been set to zero.';
    warning(sprintf([warnmess1 warnmess2])); 
    X(X<0) = 0; 
end

% POSITIVE VERSUS ZERO ELEMENTS
[m,n] = size(X);
idxP = find(X);
idx0 = setdiff(1:m*n,idxP);
normX=norm(X,'fro');

% MEAN AND VARIANCE
Xp = nonzeros(X);
meanX = sum(Xp)/(m*n);
vrncX = sum(Xp.^2)/(m*n) - meanX^2;

% INITIALIZE
idxp=(X>0);
sigmaSq = vrncX;
Z0 = zeros(m,n);   
Z0(idxp) = nonzeros(X);

%Allocate variables
Z=Z0; Theta=param.Theta0; Z_old=Z; Z_old_old=Z;

%Initialize error and time
err(1)=norm(  max(Theta,0) - X, 'fro')/normX; 
time(1)=0;

if param.display == 1
    disp('Running A-EM-NMD, evolution of [iteration number : relative error in %]');
end

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.5;
for i=1:param.maxit
    tic

  % E-STEP
  [~,Z(idx0),vrnc0] = gaussRT(Theta(idx0),sigmaSq);

  %Momentum step
  Z=Z+param.alpha*(Z_old-Z_old_old);

  % M-STEP
  [W,S,V] = tsvd(Z,r);
  H = S*V';
  Theta = W*H;
  sigmaSq = (vrnc0 + sum(sum((Theta-Z).^2)))/(m*n);
  
  %Error computation
  err(i+1) = norm( max(Theta,0) - X, 'fro')/normX; 
  if err(i+1)<param.tol 
      time(i+1)=time(i)+toc;
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

  %Stopping criteria on time
  time(i+1)=time(i)+toc;
  if time(i+1)>param.time
        break
  end

  %Update old variables
  Z_old_old=Z_old; Z_old=Z;
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
end