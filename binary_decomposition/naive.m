% function

function [Theta,err,i,time]=naive(X,r,param)

% Compute the approximate solution of the following non-linear
% where X  is binary matrix and f(theta) = (sgn(theta)+1)/2

% param settinf
[m,n]=size(X);
defaults = struct('epsilon',1e-6,'Theta', randn(m, n), 'maxit', 1000, 'tol', 1e-4, 'tolerr', 1e-5, 'time', 20,'display', 1);
if nargin < 3
    param = defaults;
else
    fields = fieldnames(param);
    for i = 1:numel(fields)
        if isfield(defaults, fields{i})
            defaults.(fields{i}) = param.(fields{i});
        end
    end
    param = defaults;
end



% latent variable Z
Z=zeros(m,n);Theta= randn(m,n);

idxn=(X==0);
idxp=(X==1);

normX=norm(X,'fro');
time(1)=0;
f= @(T) (sign(T)+1)/2;
relative_err= @(theta) norm(X-f(theta),'fro')/normX;

err(1)=relative_err(Theta);

if param.display == 1
    disp('Running A-Naive-NMD, evolution of [iteration number : relative error in %]');
end



Z_old=Z; Z_old_old=Z;

%Display setting parameters along the iterations
cntdis = 0; numdis = 0; disp_time=0.5;

for i=1:param.maxit
    tic
    % Update Z
    Z(idxn)= min(-param.epsilon,Theta(idxn));
    Z(idxp)=max(param.epsilon,Theta(idxp));
    
    %Momentum step
    Z=Z+param.alpha*(Z_old-Z_old_old);
    
    % Update theta
    [W,D,V]=tsvd(Z,r);
    Theta=W*D*V';
    
    % Error computation
    err(i+1)=relative_err(Theta);
    time(i+1)=time(i)+toc;
    
    % Stopping criteria time
    if (i>20 && abs(err(i+1)-err(i-20))<param.tolerr)
        fprintf('Algorithm has converged , minimum relative error reached =%d',[err(i+1)]);
        break
    end
    
    %Update old variables
    Z_old_old=Z_old; Z_old=Z;
    
    if param.display == 1 && mod(i,10)==0
        disp_time = min(60,disp_time*1.5);
        fprintf('[%2.2d : %2.2f] - ',i,100*err(i+1));
        cntdis = time(i+1)+disp_time; % display every disp_time
        numdis = numdis+1;
        if mod(numdis,5) == 0
            fprintf('\n');
        end
    end
    
    
end
end