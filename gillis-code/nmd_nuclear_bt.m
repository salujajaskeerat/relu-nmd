function[Theta,nuc] = nmd_nuclear_bt(X, Theta, maxiter)

% Computes an approximate solution of the following nuclear norm
% minimization problem 
%
% min_{Theta} ||Theta||_{*} such that: Theta_{ij} = X_{ij} when X_{ij} > 0
%                                      Theta_{ij} <= 0     when X_{ij} = 0                                     
%
% using the Projected Subgradient Method
%
% Update of Theta: Theta^{k+1} = PR (Theta^{k} - alpha_{k}.* (U*V') )
%
% where alpha_{k} is the adaptive stepsize updated at each iteration using
% a bactracking procedure
%
% U and V are the orthogonal matrix factors computed from the SVD of Theta
% where Theta = U*S*V' is the SVD of Theta         
%
% PR is the projection of the current iterate onto the feasible set
%
%****** Input ******
%   X       : m-by-n matrix, sparse and non negative
% Theta     : m-by-n random matrix as an initial point
% maxiter   : maximum number of iterations to be performed (set by user)

% ****** Output ******
%   Theta   : m-by-n matrix, satisfying the nuclear norm minimization
%             problem defined above (Theta has a much reduced nuclear norm) 
%   nuc     : vector keeping track of the nuclear norm of Theta at each 
%             iteration 
% 
% See the paper ''Accelerated Algorithms for Nonlinear Matrix Decomposition 
% with the ReLU function'', Giovanni Seraghiti, Atharva Awari, Arnaud 
% Vandaele, Margherita Porcelli, and Nicolas Gillis, 2023. 

[m,n] = size(X);
if min(X(:)) < 0
    warnmess1 = 'The input matrix should be nonnegative. \n';
    warnmess2 = '         The negative entries have been set to zero.';
    warning(sprintf([warnmess1 warnmess2]));
    X(X<0) = 0;
end
idxP = find(X);
idx0 = setdiff(1:m*n,idxP);
alpha=1/1^(0.1);                    %Initial choice for alpha
Theta(idxP) = X(idxP);              %Set the fixed components of Theta
Theta(idx0) = min(0, Theta(idx0));  


for i = 1: maxiter
    
    %SVD computation
    [U, D, V] = svd(Theta, 'econ');
    nuc(i)=sum(sum(D));               %Nuclear norm evaluation
    
    %Backtracking procedure
    if i>1 && nuc(i)<nuc(i-1)
        alpha=alpha*1.2;
    else
        alpha=alpha*0.7;
    end

    %Update Theta
    Theta = Theta -  (alpha* (U*V'));
    
    %Project Theta
    Theta(idxP) = X(idxP);
    Theta(idx0) = min(0, Theta(idx0));

   
end

end