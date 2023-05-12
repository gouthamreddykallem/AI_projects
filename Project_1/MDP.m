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


for i = 1:num_iterations
    % For each state in the maze
    for x = 1:size(maze,1)
        for y = 1:size(maze,2)
            % If the state is not a wall
            if maze(x,y) == -1
                % Calculate the value of the state under the current policy
                for a=1:length(actions)
                    prob(x,y,a) = action_probabilities(a) + discount * sum(sum(policy(x,y)*V));
                end
            end
        end
    end
end





for i=1:n

    for j=1:n

        
        for a = 1:length(actions)
            prob(i,j,a) = getProbs(a,maze,i,j,n,action_probabilities);
        end
        

    end
end


V = zeros(n*n, 1);
policy = zeros(n*n, 1);

% Perform model training
epoches = 1000;
for i=1:epoches
    for s = 1:n*n
        v = V(s);
        Q = zeros(1, 4);
        for a = 1:length(actions)
            [next_state, reward] = get_rewards(s, actions(a), maze, p1, p2, p3, r1, r2);
            Q(a) = reward + gamma * V(next_state);
        end
        [V(s), policy(s)] = max(Q);
    end
end


optPolicy = reshape(policy, n, n);
valFun = reshape(V, n, n);


figure(1);
subplot(1,2,1);
imagesc(maze);
title("Maze");
subplot(1,2,2);
imagesc(valFun);
title("Value function");
disp(valFun);





% subplot(1,3,3);
% imagesc(optPolicy);
% title("optimal Policy");





function [next_state, reward] = get_rewards(state, action, maze, p1, p2, p3, r1, r2)
    % Computes the next state and reward given the current state and action
        % Initialize next state and reward
        next_state = state;
        reward = 0;
        [n, ~] = size(maze);
        [row, col] = ind2sub([n, n], state);
        % Compute next state and reward for d=1
        if action == "^"     % up
            if row == 1
                next_state = state;
                reward = r2;
            else
                if col == 1 || col == n
                    next_state = sub2ind([n, n], row-1, col);
                    reward =  getReward(maze,row-1, col);                    
                else
                    r = rand();
                    if r < p1
                        next_state = sub2ind([n, n], row-1, col);
                        reward = getReward(maze,row-1, col);
                    elseif r < p1 + p2
                        next_state = sub2ind([n, n], row-1, col-1);
                        reward = getReward(maze,row-1, col-1);
                    elseif r < p1 + p2 + p3
                        next_state = sub2ind([n, n], row-1, col+1);
                        reward = getReward(maze,row-1, col+1);
                    else
                        next_state = state;
                        reward = r1;
                    end
                end
            end
            elseif action == "v" % down
                if row == n
                    next_state = state;
                    reward = r2;
                else
                    if col == 1 || col == n
                        next_state = sub2ind([n, n], row+1, col);
                        reward = getReward(maze,row+1, col);
                    else
                        r = rand();
                        if r < p1
                            next_state = sub2ind([n, n], row+1, col);
                            reward = getReward(maze,row+1, col);
                        elseif r < p1 + p2
                            next_state = sub2ind([n, n], row+1, col-1);
                            reward = getReward(maze,row+1, col-1);
                        elseif r < p1 + p2 + p3
                            next_state = sub2ind([n, n], row+1, col+1);
                            reward = getReward(maze,row+1, col+1);
                        else
                            next_state = state;
                            reward = r1;
                        end
                    end
                end
                elseif action == "<" % left
                    if col == 1
                        next_state = state;
                        reward = r2;
                    else
                        if row == 1 || row == n
                            next_state = sub2ind([n, n], row, col-1);
                            reward = getReward(maze,row, col-1);
                        else
                            r = rand();
                            if r < p1
                                next_state = sub2ind([n, n], row, col-1);
                                reward = getReward(maze,row, col-1);
                            elseif r < p1 + p2
                                next_state = sub2ind([n, n], row-1, col-1);
                                reward = getReward(maze,row-1, col-1);
                            elseif r < p1 + p2 + p3
                                next_state = sub2ind([n, n], row+1, col-1);
                                reward = getReward(maze,row+1, col-1);
                            else
                                next_state = state;
                                reward = r1;
                            end
                        end
                    end
                    elseif action == ">"
                        if col == n
                            next_state = state;
                            reward = r2;
                        else
                            if row == 1 || row == n
                                next_state = sub2ind([n, n], row, col+1);
                                reward = getReward(maze,row, col+1);
                            else
                                r = rand();
                                if r < p1
                                    next_state = sub2ind([n, n], row, col+1);
                                    reward = getReward(maze,row, col+1);
                                elseif r < p1 + p2
                                    next_state = sub2ind([n, n], row-1, col+1);
                                    reward = getReward(maze,row-1, col+1);
                                elseif r < p1 + p2 + p3
                                    next_state = sub2ind([n, n], row+1, col+1);
                                    reward = getReward(maze,row+1, col+1);
                                else
                                    next_state = state;
                                    reward = r1;
                                end
                            end
                        end
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


function [reward] = getReward(maze,i,j)
    wall = -1;
    penalty = -10;
    if(maze(i,j)==wall)
        reward = penalty;
    else
        reward = maze(i, j);
    end

end

function [prob] = getProbs(action,maze,i,j,n,action_probabilities)

if(action==1)

    if(i-1<1)
        prob = 0;
    else
        prob = checkWall(action,maze,i-1,j,action_probabilities);
    end

end


end




function [prob] = checkWall(action,maze,i,j,action_probabilities)
    wall = -1;
    penalty = 0;
    reward = 100;
    if(maze(i,j)==wall)
        prob = penalty;
    elseif(maze(i,j)==reward)
        prob=1;
    else
        prob = action_probabilities(action);
    end

end

