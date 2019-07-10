
# What data should be trained on?
If you train on the most recent n games, the agent might forget certain skills which it learned earlier.
This is because the agent no longer encounters those situtations since the situtations only appeared in the first place due to it's own actions.
This will cause the agent to encounter those situtations once again as it has forgot how to prevent them from arising, and it will get stuck in a loop of improvement and forgetting.

If instead the agent is trained on cumulative moves from the past, it improves at the beginning, but as the number of moves gets very large, the contribution of new moves becomes relatively smaller.



Was able to get good convergence for one run with GAME_COUNT=100 and a training data function of `x = x[::2] + y`
This was able to reach 80% win rate and remained around a stable 70% for many generations.

Tried a second run with a function `x = x[i for i in x if 66% chance] + y`.
This run was not able to get better than 50% win rate and remained around 25%.


