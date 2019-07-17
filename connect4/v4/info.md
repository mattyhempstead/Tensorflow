
Real Q learning

Also attempt a double headed monster
Currently achieved by simply adding an extra output to the network, being the chance of winning assuming the action in the state-action input pair is taken.

Doesn't seem to be training well? :(
Maybe this equally weighted task of predicting win probability is much harder to train and so is messing with traininf for q values.


Isn't the q values thing basically just win probability but with a discount factor?
Instead network should be taking in the state alone and predicting:
 - Move probabilities for each action
 - Win probability for the current state, rather than each potential future state

This will allow the network to learn the win probability function seperately from the strength of each move.
Currently agent is forced to learn win probaiblity give a move, which is basically just the q value anyway for an sa pair.

Will need to train network differently however.
Can use policy gradients (where a +1 or -1 gradient is given to the selected output (move) depending on win/loss, and 0 gradient to every other output)
This is similar to setting the target for each other move to its current value, and the target for the selected move to be +/- 1. Then training with these targets.
I feel like the first method will train faster as the network gets better, since the gradients will be a constant value.
However the second method might train faster while the network is very inaccurate and gets big errors?


A single 1500 long run was made, results were very good and network seems to achieve higher than the 2000 run made without two headed monster technique.
