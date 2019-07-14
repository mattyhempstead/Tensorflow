import numpy as np
import random, sys
import selectMove

def playTrainingGames(n, game, model, opponent, d=0.9):
    pastBoardRewards = []
    pastBoardStates = []
    winners = []

    for k in range(n):

        # Used to keep track of the number of states encountered in current game
        stateCount = 0  

        while True:
            if game.turn == 1:
                moveRanks = getMoveRanks(game, model)
                move = selectMove.maxMove(moveRanks)
                winner = game.move(move)

                # Add all states except first one
                if (game.turnNum > 0):
                    pastBoardStates.append(game.board.copy())
                    stateCount += 1

            else:
                winner = opponent.playMove()

            if winner != None:
                break

        winners.append(winner)

        # Set reward for each state
        # This is the same as using +1 for win and -1 for loss, except scaled between 0 and 1
        # Calculated using discount factor (g), and number of moves until game end (n)
        # Win = (1+g^n)/2, Loss = (1-g^n)/2
        # For each move in game up until final move
        for i in range(stateCount):
            reward = d ** (stateCount - i - 1)
            if winner == 1:
                reward = (1 + reward) / 2
            else:
                reward = (1 - reward) / 2
            
            pastBoardRewards.append([reward])

        # print(len(pastBoardStates), pastBoardRewards)
        # sys.exit()

        # game.printGame()
        # print("winner #{}: {}".format(k, winner))
        print("Game #{}/{}".format(k+1,n), end='\r')

        game.resetGame()
        stateCount = 0

    return {
        "pastBoardRewards": np.array(pastBoardRewards),
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


# def getMoveRanks(game, model):
#     moveRanks = np.zeros(7, dtype=float)

#     for i in range(7):
#         moveIndex = game.getValidMove(i)
#         if moveIndex == -1:
#             moveRanks[i] = 0
#         else:
#             game.placePiece(i,moveIndex)

#             boardInput = np.array([game.getInput()])
#             q = model.predict(boardInput)[0][0]
#             moveRanks[i] = q

#             game.removePiece(i,moveIndex)

#     game.printGame()
#     print(moveRanks)
#     input()

#     return moveRanks







# from connect4 import Connect4Game, RandomAgent, GoodAgent

# def playTrainingGames(n, game, model, opponent, d=0.9):
#     pastBoardRewards = []
#     pastBoardStates = []
    

#     games = [Connect4Game() for i in range(n)]
#     winners = [None for i in range(n)]

#     states = [[] for i in range(n)]


#     # Have a list of games and each game holds a list of past states
#     # Generate a list of current states 



#     for k in range(n):

#         # Used to keep track of the number of states encountered in current game
#         stateCount = 0  

#         while True:
#             if game.turn == 1:
#                 moveRanks = getMoveRanks(game, model)
#                 move = selectMove.maxMove(moveRanks)
#                 winner = game.move(move)

#                 # Add all states except first one
#                 if (game.turnNum > 0):
#                     pastBoardStates.append(game.board.copy())
#                     stateCount += 1

#             else:
#                 winner = opponent.playMove()

#             if winner != None:
#                 break

#         winners.append(winner)

#         # Set reward for each state
#         # This is the same as using +1 for win and -1 for loss, except scaled between 0 and 1
#         # Calculated using discount factor (g), and number of moves until game end (n)
#         # Win = (1+g^n)/2, Loss = (1-g^n)/2
#         # For each move in game up until final move
#         for i in range(stateCount):
#             reward = d ** (stateCount - i - 1)
#             if winner == 1:
#                 reward = (1 + reward) / 2
#             else:
#                 reward = (1 - reward) / 2
            
#             pastBoardRewards.append([reward])

#         # print(len(pastBoardStates), pastBoardRewards)
#         # sys.exit()

#         # game.printGame()
#         # print("winner #{}: {}".format(k, winner))
#         print("Game #{}/{}".format(k,n), end='\r')

#         game.resetGame()
#         stateCount = 0

#     return {
#         "pastBoardRewards": np.array(pastBoardRewards),
#         "pastBoardStates": np.array(pastBoardStates),
#         "winners": winners
#     }







