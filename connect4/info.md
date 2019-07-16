
Have agent play a game against a random-move agent
Throughout game, agent predicts Q value for each state
After game, train agent to output correct Q value for the whole game (1 if win, 0 if loss)
This will cause agent to converge to outputting the probability of winning a particular game


# Working training
300 games between train
train for full epoch on data
repeat and add to data pool


# Greedy rewards
Should agent be trained to output 1 for all moves in a winning game?
Maybe instead it should be trained to output `1 * e^n` where n is the number of moves until it wins and `0 < e < 1`
This however would not probagate winning probabilities correctly, as even if the agent always wins after say 10 moves, they would output a probability of winning less than one.
Another alternative is to learn to output the next predicted Q-value, and if the agent is playing the final winning or losing move, they are trained to output 1 and 0 respectively. This would cause these probabilities to slowly propagate backwards from high-certainty moves near the end to the beginning.
Should definitely try both alternatives, and look up some more online.


# Potential improvement
Have agent play many games simultaneously to for fast 



## Proof that the outputted Q-value will converge to the probability of winning

The cost function for the outputted Q-value if the agent wins is defined as:
    `W(x) = 1*ln(x) + (1-1)*ln(1-x) = ln(x)`

The cost function for the outputted Q-value if the agent loses is defined as:
    `L(x) = 0*ln(x) + (1-0)*ln(1-x) = ln(1-x)`

Define the probability of winning in a particular state as 0 < P < 1
The expected cost in a single game for the agent assuming they output a Q-value of x is thus:
    `C(x) = P*W(x) + (1-P)*L(x) = P*ln(x) + (1-P)*ln(x)`
This is simply a binary cross-entropy loss function, and thus has a minimum at x=P

Thus the agent will converge to outputting the probability of winning as their Q-value for this state
as it is the minimum value for the cost function which is being optimised.




# What data should be trained on?
If you train on the most recent n games, the agent might forget certain skills which it learned earlier.
This is because the agent no longer encounters those situtations since the situtations only appeared in the first place due to it's own actions.
This will cause the agent to encounter those situtations once again as it has forgot how to prevent them from arising, and it will get stuck in a loop of improvement and forgetting.

If instead the agent is trained on cumulative moves from the past, it improves at the beginning, but as the number of moves gets very large, the contribution of new moves becomes relatively smaller.



Was able to get good convergence for one run with GAME_COUNT=100 and a training data function of `x = x[::2] + y`
This was able to reach 80% win rate and remained around a stable 70% for many generations.
Reached a loss of ~0.41 with 3000 training samples per train.

Tried a second run with a function `x = x[i for i in x if 66% chance] + y`.
This run was not able to get better than 50% win rate and remained around 25%.



# Network goal

Currrently network outputs a Q-value ranking each of the possible future states.
Highest value is selected.
This limits the networks exploration possibilities.

Should instead have network select move based on probabilities of move outputs.
After game, train for played moves to have Q value 1 or 0.


Network outputs policy 


Network predicts future state.
Network rates future state to select move.




# Training issue
I am currently training the bots to output the Q value predicted by the next state.
This means that after the bot has been trained to improve, these Q values might no longer be relevant and
the bot might actually be training to get worse.

e.g. At one point in training, the agent wins going from state A -> B -> C
The Q values outputted during this game are:
Q(A) = 0.5, Q(B) = 0.6, Q(C) = 0.7

Later in the simulation, after the agent has been trained to be objectively better, it now outputs
Q(A) = 0.7, Q(B) = 0.8, Q(C) = 0.9

If the agent was to be trained on this previous win, it would be trained such that
Q(A) -> 0.6, Q(B) -> 0.7, Q(C) -> 1
This would make the agent objectively worse.



The ultimate aim is to make it so that moves which lead to winning should be encouraged to play.
Moves which lead to losing should be discouraged to play.

If the agent is trained based on the decaying reward functionality, will a similar issue happen?
Since the moves are selected relative to the Q values outputted for alternate moves, if an agent is
encouraged to output a move more than it is encouraged to output the alternate moves, it will learn
to select this move.
e.g. An agent might only be trained to output 0.1 and 0.15 for a move near the beginning, 
yet it will still learn to select the second move over the first.

This method will also encourage winning quicker.



# Actor Critic
Actor network generates a policy
Critic network produces value this policy?



Rather than storing past predictions for Q values, I should instead store only the state-action-states.
The new states don't need to have their Q-values updated.
The old states have old predictions of Q-values.

Network predicts a Q-value for a state and selects an actions.

Any one game has a chain of states and predicted Q-values which led to that state.
If the agent have only recently played each of those games, the Q-values between states will all be the same.
However, if the agent is recalling these states from memory, the Q-values which they used to produce this chain of states will all be different.

This method does not work with simply a value function for states.
It is specifically for 

Agent predicts a Q-value for a state-action pair.
The agent is then trained on the actual Q-value for the state-action pair.
This Q-value should be the immediate reward (zero) plus the maximum Q-value for all actions in the future state it ends up in.

This is only for situtations where the agent 



# Real Q-value

Agent takes in action-state pairs.
Predicts the Q-value for each of these action-state pairs and the largest Q-value is used to select the move it will make.
For explorations purposes, a system could be implemented where the action with the largest Q-value is not always used, instead either a random move could be selected with probability p, or a similar system based on the rankings of Q-values could be used.


For each move made, we store the state-action pair and the next state reached from this state-action pair.
We should also store the Q-value which led to the selection of this state-action pair, however these are only used for the next training run and nothing else.

For each s-a-s' triplet, find the maximum Q(s', a') by testing all possible a'
This will require performing a full prediction for each move obtained from memory.

The network is then trained such that Q(s, a) <- g*max_a'_Q(s', a')
'g' refers to the discount factor to encourage obtaining rewards quicker (i.e. winning quicker)
If s' is a game ending state, train Q(s, a) <- (1 if win, 0 if loss) 


Train network and store s-a-s' and also an extra array containin Q for each s-a
Select s-a-s' triplets from the past, and calculate the g*max_a'_(Q(s',a')) for the s'
Append these max Q's to the Q array with their respective s-a-s'
Train network to output values from the Q array given the respective s-a


This will take a long time to compute Q-values for past memories depending on the proportion of current states vs past states


# Extra training

Instead of training with recent experience, have a memory bank of moves
A batch of these moves should then be selected at random for training.
This memory will need to be flushed at times to prevent memory leak.
Maybe store up to 10000 memories, and every extra memory will randomly replace one of them.


Now we train the agent to predict other values:

Win percentage
 - Split the network at the end and feed only the state to predict a single number win %

Predict opponent
 - Network tries to predict the next state it will encounter

