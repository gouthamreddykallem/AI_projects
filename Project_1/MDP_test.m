T = readtable('map.csv');
maze=T{:,:}; %Convert map into Matrix.

n=size(maze,1);

[robotLocation,destinationLocation]=findSourceAndTraget(maze); % Retrive the Robot location and Destination location

disp(destinationLocation);

disp(maze);


gamma = 0.9;    % discount factor
penalty = -10;  % penalty for hitting a wall


actions = ["^","v","<",">","^^","vv","<<",">>"];
action_probabilities = [0.2 0.2 0.2 0.3 0.25 0.2 0.15 0.3];

stay = 0.1;

r1 = 1;         % reward for taking a step (d=1)
r2 = 2;         % reward for taking a step (d=2)

prob = zeros(n,n,length(actions));
policy = ones(size(maze));
discount = 0.99;
num_iterations = 100;

% for i = 1:num_iterations
%     % For each state in the maze
%     for x = 1:size(maze,1)
%         for y = 1:size(maze,2)
%             % If the state is not a wall
%             if maze(x,y) ~= -1
%                 % Calculate the value of the state under the current policy
%                 for a=1:length(actions)
%                     prob(x,y,a) = action_probabilities(a) + discount * sum(sum(policy(x,y)*prob(:,:,a)));
%                 end
%             end
%         end
%     end
% end




for i=1:n
 
     for j=1:n
 
         
         for a = 1:length(actions)
             prob(i,j,a) = getProbs(a,maze,i,j,n,action_probabilities);
         end
         
 
     end
 end





function [prob] = getProbs(action,maze,i,j,n,action_probabilities)

    if(action==1)
        if(i-1<1)
            prob = 0;
        else
            prob = checkWall(action,maze,i-1,j,action_probabilities);
        end

    elseif(action==2)
        if(i+1>n)
            prob = 0;
        else
            prob = checkWall(action,maze,i+1,j,action_probabilities);
        end

    elseif(action==3)
        if(j-1<1)
            prob = 0;
        else
            prob = checkWall(action,maze,i,j-1,action_probabilities);
        end

    elseif(action==4)
        if(j+1>n)
            prob = 0;
        else
            prob = checkWall(action,maze,i,j+1,action_probabilities);
        end


    elseif(action==5)
        if(i-2<1)
            prob = 0;
        else
            prob = checkWall(action,maze,i-2,j,action_probabilities);
        end

    elseif(action==6)
        if(i+2>n)
            prob = 0;
        else
            prob = checkWall(action,maze,i+2,j,action_probabilities);
        end

    elseif(action==7)
        if(j-2<1)
            prob = 0;
        else
            prob = checkWall(action,maze,i,j-2,action_probabilities);
        end

    elseif(action==8)
        if(j+2>n)
            prob = 0;
        else
            prob = checkWall(action,maze,i,j+2,action_probabilities);
        end


    else 
        prob =0.1;
    
    end


end




function [prob] = checkWall(action,maze,i,j,action_probabilities)
    wall = -1;
    penalty = 0;
    reward = 100;
    n=size(maze);
    if(i>n | j>n)
        prob = 0;
    elseif(maze(i,j)==wall)
        prob = penalty;
    elseif(maze(i,j)==reward)
        prob=1;
    else
        prob = action_probabilities(action);
    end

end



function [robotLocation,destinationLocation] = findSourceAndTraget(maze)
    n=size(maze,1);
    for i=1:n
        for j=1:n
            if(maze(i,j)==10)
                robotLocation=[i,j];
            end
            if(maze(i,j)==100)
                destinationLocation=[i,j];
            end
        end
    end
end
