% Define total number of nodes
num_nodes=1005;

adj_matrix=zeros(num_nodes);

% read the graph dataets

fid=fopen('email-Eu-core.txt','r');


while ~feof(fid)
    line = fgetl(fid);

    if isempty(line)
        continue;
    end
    
    % Split the line into individual numbers
    edge_pairs = str2num(line); %#ok<*ST2NM>
    
    % Check if two numbers were successfully extracted
    if numel(edge_pairs) == 2
        % Extract u and v from the edge_pairs
        u = edge_pairs(1);
        v = edge_pairs(2);
        
        % Display edge pairs
        fprintf('Edge pair: u = %d, v = %d\n', u, v);
    else
        disp('Invalid edge pair format');
    end


    % For directed graph . change accordingly for undirected graph
    adj_matrix(u+1,v+1)=1; 
end

% close file

fclose(fid);

% save the adj_matrix
save('email-Eu-core.mat','adj_matrix');