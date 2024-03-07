


function [Theta,err,i,time,Z,alpha_it]=anderson(X,rank,param)
    
    [m, n] = size(X);
    defaults = struct('Theta0', randn(m, n), 'maxit', 1000, 'freeze',10,'tol', 1e-4, 'tolerr', 1e-5, 'time', 20,'depth',10, 'display', 1,'C',10,'gaurded',false);
    % param descriptions
    % param.freeze : The anderson acceleration is not applies for first freeze iterations
    % Note : param.freeze >=param.depth
    
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



    %Detect (negative and) positive entries of X
    if min(X(:)) < 0
        warnmess1 = 'The input matrix should be nonnegative. \n';
        warnmess2 = '         The negative entries have been set to zero.';
        warning(sprintf([warnmess1 warnmess2]));
        X(X<0) = 0;
    end
    [m,n]=size(X);
    normX=norm(X,'fro');idx=(X==0);idxp=(X>0);

    %Initialize the latent variable
    Z0 = zeros(m,n);Z0(idxp) = nonzeros(X);

    %Create istances for variables
    Z=Z0; Theta=param.Theta0;z=Z0(:); %intialize the z 

    f= @(theta_old) min(0,theta_old.*idx) + X.*idxp;

    %Initialize error and time counter
    err(1)=norm(max(0,Theta)-X,'fro')/normX;
    time(1)=0;

    % f stores the fz^i while R (residual matrix) stores the r^i = f^i - z
    F=[];R=[];
    alpha_it=[ones(param.depth,1)];

    %Display setting parameters along the iterations
    ones_d=ones(param.depth,1);
    cntdis = 0; numdis = 0; disp_time=0.1;


    % Start the iterations
    for i=1:param.maxit
        tic
        %Update on Z ---> Z=min(0,Theta)
        fZ=f(Theta);fz=fZ(:);r=fz-z;
        F=stack_and_truncate(F,fz,param.depth);
        R=stack_and_truncate(R,r,param.depth);
        
        acclerated=false;
        if(i>param.freeze && i>param.depth)

            % Minimize |R*alpha| . Find alphs
            alpha= R'*R \ ones_d;
            alpha=alpha/ (ones_d' *alpha);

            % z^(i+1) = F^(i)alpha
            % Z^(i+1) = matrix(z)
            alpha_it = [alpha_it, alpha(:)];
            z=F*alpha;
            acclerated=true;
        else
            z=fz;
        end
        Z=reshape(z,m,n);
        % Anderson-step to compare 
        
        %Update of T
        [W,D,V] = tsvd(Z,rank);  %function computing TSVD
        Theta=W*D*V';
        
        %Error computation
        Ap=max(0,Theta);
        
        err(i+1)=norm(Ap-X,'fro')/normX; 
        if(param.gaurded && err(i+1)>err(i) && acclerated)
            % error did'nt improve
            z=fz;
            Z=reshape(z,m,n);
            [W,D,V] = tsvd(Z,rank);  %function computing TSVD
            Theta=W*D*V';
            Ap=max(0,Theta);
            err(i+1)=norm(Ap-X,'fro')/normX; 
        end
        %Standard stopping condition on the relative error
        if err(i+1)<param.tol
            time(i+1)=time(i)+toc; %needed to have same time components as iterations
            if param.display == 1
                if mod(numdis,5) > 0, fprintf('\n'); end
                fprintf('The algorithm has converged: ||X-max(0,WH)||/||X|| < %2.0d\n',param.tol);
            end
            break
        end
        if i >= 20  &&  abs(err(i+1) - err(i-19)) < param.tolerr
            time(i+1)=time(i)+toc; %needed to have same time components as iterations
            if param.display == 1
                if mod(numdis,5) > 0, fprintf('\n'); end
                fprintf('The algorithm has converged: rel. err.(i+1) - rel. err.(i+10) < %2.0d\n',param.tolerr);
            end
            break
            break
        end
        
      
       
        
        %Stopping condition on time
        time(i+1)=time(i)+toc;
        
        
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

function stacked_matrix = stack_and_truncate(F, z, depth)
    % Determine the number of columns in F
    num_cols_F = size(F, 2);

    % Calculate the number of columns to keep after stacking z
    num_cols_keep = min(num_cols_F + 1, depth);

    % Stack z onto F column-wise and keep only the last 'num_cols_keep' columns
    stacked_matrix = [F(:, end - num_cols_keep + 2:end), z];
end
