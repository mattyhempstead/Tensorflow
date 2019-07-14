import numpy as np
import random
# from selectMove import selectMoveMax, selectMoveOdds
import selectMove

def playTrainingGames(n, game, model, opponent):
    pastBoardQValues = []
    pastBoardStates = []
    winners = []


    for k in range(n):

        while True:
            if game.turn == 1:
                moveRanks = getMoveRanks(game, model)
                move = selectMove.maxMove(moveRanks)
                winner = game.move(move)

                # Add q value for move unless first move
                if (game.turnNum > 0):
                    pastBoardQValues.append([moveRanks[move]])
                    pastBoardStates.append(game.board.copy())

            else:
                winner = opponent.playMove()

            if winner != None:
                break

        if winner == 1:
            pastBoardQValues[-1] = [1]
        else:
            pastBoardQValues[-1] = [0]

        winners.append(winner)
        
        # game.printGame()
        # print("winner #{}: {}".format(k, winner))
        print("Game #{}/{}".format(k,n), end='\r')

        game.resetGame()

    return {
        "pastBoardQValues": np.array(pastBoardQValues),
        "pastBoardStates": np.array(pastBoardStates),
        "winners": winners
    }



def getMoveRanks(game, model):
    '''
        Gets the move ranks for each of the valid moves.
        Valid moves are all predicted in one model.predict() call as that is faster to compute.
    '''

    moveRanks = np.zeros(7, dtype=float)

    states = []

    validMoves = [i for i in range(7) if (game.isValidMove(i))]
    for i in validMoves:
        game.placeCol(i)
        states.append(game.getInput())
        game.removeCol(i)

    states = np.array(states)
    predictions = model.predict(states)

    for i in range(7):
        if i in validMoves:
            moveRanks[i] = predictions[validMoves.index(i)][0]

    return moveRanks

