import numpy as np
import random, sys

def playTrainingGames(n, game, model, opponent, discountFactor=0.9, exploit=1):
    '''
        Plays n games between model and opponent. \n
        Returns the data of all the games which can be used for training. \n 
        
        exploit -- The chance of selecting chosen move (complement of selecting random move)
    '''

    # Each move in each game adds an element to these arrays
    saPairs = []
    sPrimes = []
    qValues = []    # These are discarded after network once stored in memory

    # One element per game is added to this array
    winners = []

    for k in range(n):

        while True:
            if game.turn == 1:

                move = game.getMove(model, exploit)
                winner = game.move(move["move"])

                # Add move to memory
                saPairs.append(move["sa"])

                # Add qValue for next move to memory
                if game.turnNum > 1:
                    qValues.append(
                        move["qValue"] * discountFactor
                    )

                    # Append the result state from the previous move
                    sPrimes.append(move["saArray"])


                        
            else:
                winner = opponent.playMove()

            if winner != None:
                if winner == 1:
                    qValues.append(np.array([1]))
                    sPrimes.append(1)
                else:
                    qValues.append(np.array([0]))
                    sPrimes.append(0)
                break

        winners.append(winner)


        # sys.exit()

        # game.printGame()
        # print("winner #{}: {}".format(k, winner))
        print("Game #{}/{}".format(k+1,n), end='\r')

        game.resetGame()

    return {
        "saPairs": np.array(saPairs),
        "sPrimes": np.array(sPrimes),
        "qValues": np.array(qValues),
        "winners": winners
    }
