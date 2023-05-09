T = readtable('map.csv');
maze=T{:,:}; %Convert map into Matrix.

n=size(maze,1);

[robotLocation,destinationLocation]=findSourceAndTraget(maze); % Retrive the Robot location and Destination location

disp(destinationLocation);

disp(maze);


gamma = 0.9;    % discount factor
epsilon = 0.001;  % stopping criterion
reward1 = 100;  % reward for reaching the goal
reward2 = -10;  % penalty for hitting a wall
p1 = 0.7;       % probability of sliding in first direction (d=1)
p2 = 0.1;       % probability of sliding in second direction (d=1)
p3 = 0.1;       % probability of sliding in third direction (d=1)
p4 = 0.1;       % probability of staying in same cell (d=1)
q1 = 0.5;       % probability of sliding in first direction (d=2)
q2 = 0.1;       % probability of sliding in second direction (d=2)
q3 = 0.1;       % probability of sliding in third direction (d=2)
q4 = 0.2;       % probability of sliding in fourth direction (d=2)
r1 = -1;        % reward for taking a step (d=1)
r2 = -2; 


V = zeros(n*n, 1);
policy = zeros(n*n, 1);

% Perform value iteration
delta = inf;
epoches = 1000;
while epoches > 0
    delta = 0;
    for s = 1:n*n
        v = V(s);
        Q = zeros(1, 4);
        for a = 1:4
            [next_state, reward] = get_transition_prob(s, a, maze, p1, p2, p3, r1, r2);
            Q(a) = reward + gamma * V(next_state);
        end
        [V(s), policy(s)] = max(Q);
        delta = max(delta, abs(v - V(s)));
    end
%     if(delta > epsilon)
%          break;
%      end
    epoches = epoches - 1;
end

% Print policy
disp("Optimal policy:");
disp(reshape(policy, n, n)');
optPolicy = reshape(policy, n, n)';
% Print value function
disp("Value function:");
disp(reshape(V, n, n)');
valFun = reshape(V, n, n)';

figure(1);
subplot(1,3,1);
imagesc(maze);
title("Maze");
subplot(1,3,2);
imagesc(optPolicy);
title("optimal Policy");
subplot(1,3,3);
imagesc(valFun);
title("Value function");




function [next_state, reward] = get_transition_prob(state, action, maze, p1, p2, p3, r1, r2)
    % Computes the next state and reward given the current state and action
        % Initialize next state and reward
        next_state = state;
        reward = 0;
        [n, ~] = size(maze);
        [row, col] = ind2sub([n, n], state);
        % Compute next state and reward for d=1
        if action == 1     % up
            if row == 1
                next_state = state;
                reward = r2;
            else
                if col == 1 || col == n
                    next_state = sub2ind([n, n], row-1, col);
                    reward = maze(row-1, col);
                else
                    r = rand();
                    if r < p1
                        next_state = sub2ind([n, n], row-1, col);
                        reward = maze(row-1, col);
                    elseif r < p1 + p2
                        next_state = sub2ind([n, n], row-1, col-1);
                        reward = maze(row-1, col-1);
                    elseif r < p1 + p2 + p3
                        next_state = sub2ind([n, n], row-1, col+1);
                        reward = maze(row-1, col+1);
                    else
                        next_state = state;
                        reward = r1;
                    end
                end
            end
            elseif action == 2 % down
                if row == n
                    next_state = state;
                    reward = r2;
                else
                    if col == 1 || col == n
                        next_state = sub2ind([n, n], row+1, col);
                        reward = maze(row+1, col);
                    else
                        r = rand();
                        if r < p1
                            next_state = sub2ind([n, n], row+1, col);
                            reward = maze(row+1, col);
                        elseif r < p1 + p2
                            next_state = sub2ind([n, n], row+1, col-1);
                            reward = maze(row+1, col-1);
                        elseif r < p1 + p2 + p3
                            next_state = sub2ind([n, n], row+1, col+1);
                            reward = maze(row+1, col+1);
                        else
                            next_state = state;
                            reward = r1;
                        end
                    end
                end
                elseif action == 3 % left
                    if col == 1
                        next_state = state;
                        reward = r2;
                    else
                        if row == 1 || row == n
                            next_state = sub2ind([n, n], row, col-1);
                            reward = maze(row, col-1);
                        else
                            r = rand();
                            if r < p1
                                next_state = sub2ind([n, n], row, col-1);
                                reward = maze(row, col-1);
                            elseif r < p1 + p2
                                next_state = sub2ind([n, n], row-1, col-1);
                                reward = maze(row-1, col-1);
                            elseif r < p1 + p2 + p3
                                next_state = sub2ind([n, n], row+1, col-1);
                                reward = maze(row+1, col-1);
                            else
                                next_state = state;
                                reward = r1;
                            end
                        end
                    end
                    elseif action == 4
                        if col == n
                            next_state = state;
                            reward = r2;
                        else
                            if row == 1 || row == n
                                next_state = sub2ind([n, n], row, col+1);
                                reward = maze(row, col+1);
                            else
                                r = rand();
                                if r < p1
                                    next_state = sub2ind([n, n], row, col+1);
                                    reward = maze(row, col+1);
                                elseif r < p1 + p2
                                    next_state = sub2ind([n, n], row-1, col+1);
                                    reward = maze(row-1, col+1);
                                elseif r < p1 + p2 + p3
                                    next_state = sub2ind([n, n], row+1, col+1);
                                    reward = maze(row+1, col+1);
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