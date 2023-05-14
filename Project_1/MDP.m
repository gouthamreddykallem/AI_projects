close all;
clear all;
clc;

% Define the environment parameters

T = readtable('map.csv');
maze=T{:,:}; %Convert map into Matrix.

n=size(maze,1);

[robotLocation,destinationLocation]=findSourceAndTraget(maze); % Retrive the Robot location and Destination location


reward_high = 10; % High reward for reaching the goal
reward_empty_space = 2;
reward_low = 0; % Low reward for hitting a wall

% Define the action parameters
d1 = 1; % Distance for action type 1
d2 = 2; % Distance for action type 2

% Define the transition probabilities


actions = ["^","v","<",">","^^","vv","<<",">>"];
action_probabilities = [0.5 0.2 0.3 0.4 0.25 0.2 0.15 0.3];



% Define the transition function
P = zeros(n, n, 8);
for i = 1:n
    for j = 1:n
        for a = 1:8
            P(i, j, a) = getProbs(a,maze,i, j,n,action_probabilities);
        end
    end
end



R = zeros(n,n);

for i = 1:n
    for j = 1:n
        if(checkWallOnly(maze,i,j)==1)
            R(i,j)=reward_low;
        elseif([i,j]==destinationLocation)
            R(i,j)=reward_high;
        else
            R(i,j)=reward_empty_space;

        end
    end
end


% Define the value function
V = zeros(n, n);




% Define the discount factor
gamma = 0.9;

% Define the value iteration parameters
epsilon = 0.01;
delta = inf;

% Run the value iteration algorithm
while delta > epsilon
    delta = 0;
    for i = 1:n
        for j = 1:n
            if [i, j] == destinationLocation
                V_new = reward_high;
            elseif maze(i, j) == -1
                V_new = reward_low;
            else
                V_new = -inf;
                for a = 1:length(actions)
                    [i_next, j_next] = get_next_state(i, j, a, n,d1,d2);
                    V_new = max(V_new, R(i,j) + gamma*sum(P(i, j, a)*V(i_next, j_next)));
                end
            end
            delta = max(delta, abs(V(i, j) - V_new));
            V(i, j) = V_new;
        end
    end
end






V(destinationLocation(1),destinationLocation(2)) = 100;
% Find the path to the high reward position using the value function
pos = robotLocation; % starting position
path = [pos];
i=2;
while ~(pos(1)==destinationLocation(1) & pos(2) == destinationLocation(2)) % until we reach the high reward position
    % Calculate the expected future rewards of all possible actions
%     up = V(pos(1)-1,pos(2));
    if pos(1) > 1
        up = V(pos(1)-1,pos(2));
    else
        up = -inf; 
    end
%     down = V(pos(1)+1,pos(2));
    if pos(1) < size(V,1)
        down = V(pos(1)+1,pos(2));
    else
        down = -inf; 
    end
    if pos(2) > 1
        left = V(pos(1),pos(2)-1);
    else
        left = -inf; 
    end
%     right = V(pos(1),pos(2)+1);
    if pos(2) < size(V,2)
        right = V(pos(1),pos(2)+1);
    else
        right = -inf; 
    end
    % Choose the action that leads to the state with the highest value
    [~,action] = max([up,down,left,right]);
    disp(action);
    % Move to the new position
    if action == 1
        pos(1) = pos(1)-1;
    elseif action == 2
        pos(1) = pos(1)+1;
    elseif action == 3
        pos(2) = pos(2)-1;
    else
        pos(2) = pos(2)+1;
    end
    disp(pos);
    % Add the new position to the path
    path = [path; pos];
end

% Show the maze and the path using imagesc
figure
figure(1);
subplot(1,3,1);
imagesc(maze);
title("Maze");
subplot(1,3,2);
imagesc(V);
title("Value function");
subplot(1,3,3);
imagesc(maze)
title("Path followed");
hold on
plot(path(:,2),path(:,1),'r','LineWidth',2)






% Define a function to get the next state given an action
function [i_next, j_next] = get_next_state(i, j, a, n,d1,d2)
    if a == 1 % Up
        i_next = max(1, i-d1);
        j_next = j;
    elseif a == 2 % Down
        i_next = min(n, i+d1);
        j_next = j;
    elseif a == 3 % Left
        i_next = i;
        j_next = max(1, j-d1);
    elseif a == 4 % Right
        i_next = i;
        j_next = min(n, j+d1);
    elseif a == 5 % Up-2
        i_next = max(1, i-d2);
        j_next = j;
    elseif a == 6 % Down-2
        i_next = max(1, i+d2);
        j_next = j;
    elseif a == 7 % Left-2
        i_next = i;
        j_next = max(1, j-d2);
    elseif a == 8 % Right-2
        i_next = i;
        j_next = min(n, j+d2);
    end
    if(i_next >15)
        i_next =15;
    end

    if(j_next >15)
        j_next =15;
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



% 
% 
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
    n=size(maze,1);
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


function [check] = checkWallOnly(maze,i,j)
    wall = -1;
    check = 0;
    if(maze(i,j)==wall)
        check = 1;
    end

end