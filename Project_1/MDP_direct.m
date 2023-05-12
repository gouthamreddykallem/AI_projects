close all;
clear all;
clc;

% Define the environment parameters

T = readtable('map.csv');
maze=T{:,:}; %Convert map into Matrix.

n=size(maze,1);

[robotLocation,destinationLocation]=findSourceAndTraget(maze); % Retrive the Robot location and Destination location


start = [1, 1]; % Starting position
reward_high = 1; % High reward for reaching the goal
reward_low = 0; % Low reward for hitting a wall

% Define the action parameters
d1 = 1; % Distance for action type 1
d2 = 2; % Distance for action type 2
r1 = -1; % Reward for taking action type 1
r2 = -2; % Reward for taking action type 2

% Define the transition probabilities


r = [0.2 0.2 0.2 0.3 0.25 0.2 0.15 0.3];

actions = ["^","v","<",">","^^","vv","<<",">>"];
action_probabilities = [0.2 0.2 0.2 0.3 0.25 0.2 0.15 0.3];

P = zeros(n, n, 8, n, n); % 8 possible actions
for i = 1:n
     for j = 1:n
         % Action type 1
         for a=1:8
             if(a==1)
                 P(i, j, 1, max(1, i-d1), j) = 0.7;
                 P(i, j, 1, max(1, i-d1), max(1, j-1)) = 0.1;
                 P(i, j, 1, max(1, i-d1), min(n, j+1)) = 0.1;
                 P(i, j, 1, max(1, i-d2), j) = 0.5;
                 P(i, j, 1, max(1, i-d2), max(1, j-1)) = 0.1;
                 P(i, j, 1, max(1, i-d2), min(n, j+1)) = 0.1;

% % %              elseif(action==2)

             end
             % Action type 2
            
             P(i, j, 2, max(1, i-d1), j) = 0.2;
             P(i, j, 2, i, j) = 0.1;%
             


             P(i, j, 1, i, j) = 0.1;
         end
    end
 end


% for i=1:n
%  
%      for j=1:n
%  
%          
%          for a = 1:8
%              P(i,j,a,i,j) = getProbs(a,maze,i,j,n,r);
%          end
%          
%  
%      end
%  end





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
                for a = 1:8
                    [i_next, j_next] = get_next_state(i, j, a, n,d1,d2);
                    V_new = max(V_new, r(a) + gamma*sum(P(i, j, a, i_next, j_next)*V(i_next, j_next)));
                end
            end
            delta = max(delta, abs(V(i, j) - V_new));
            V(i, j) = V_new;
        end
    end
end


figure(1);
subplot(1,2,1);
imagesc(maze);
title("Maze");
subplot(1,2,2);
imagesc(V);
title("Value function");
% disp(V);


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


