
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



