import numpy as np

def playTrainingGames(n, game, model, opponent):
    pastBoardQValues = []
    pastBoardStates = []
    winners = []

    for k in range(n):

        while True:
            if game.turn == 1:
                moveRanks = getMoveRanks(game, model)
                move = moveRanks.argmax()
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
    moveRanks = np.zeros(7, dtype=float)

    for i in range(7):
        moveIndex = game.getValidMove(i)
        if moveIndex == -1:
            moveRanks[i] = 0
        else:
            game.placePiece(i,moveIndex)

            boardInput = np.array([game.getInput()])
            q = model.predict(boardInput)[0][0]
            moveRanks[i] = q

            game.removePiece(i,moveIndex)

    return moveRanks

