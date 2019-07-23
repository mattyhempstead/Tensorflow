import sys, math, random, numpy as np

class Connect4Game():
    def __init__(self):

        self.turn = 1
        self.turnNum = 0	# Keeps track of number of turns played (useful for detecting game over)

        self.board = np.zeros((7,6), dtype="int8")

    def __repr__(self):
        '''
            Returns a string repr of the board state
        '''
        board = np.array([[j for j in i][::-1] for i in self.board])
        board = np.transpose(board)

        string = ""
        string += "\n"
        for row in board:
            for col in row:
                string += " X " if col==1 else (" O " if col==-1 else " - ")
            string += "\n"
        string += " 1  2  3  4  5  6  7 "
        string += "\n"
        return string

    def printGame(self):
        '''
            Prints a basic version of the board state to stdout
        '''
        print(self)

    def move(self, col):	
        '''
            Makes move in a particular column and switches turn \n
            If move ends game, winner is returned (1 or 0 or -1) \n
            Otherwise returns None \n
            If move is not valid, prints to stdout \n
        '''
        for row in range(6):
            if self.board[col][row] == 0:

                self.board[col][row] = self.turn
                self.turnNum += 1			

                winner = self.doesMoveEndGame(col, row, self.turn)

                if winner == None:
                    self.turn *= -1
                
                return winner

        print("Invalid move")

    def placePiece(self, col, row):
        self.board[col][row] = self.turn

    def removePiece(self, col, row):
        self.board[col][row] = 0

    def placeCol(self, col):
        row = self.getValidMove(col)
        self.placePiece(col,row)
    
    def removeCol(self, col):
        for i in range(5,-1,-1):
            if self.board[col][i] != 0:
                self.board[col][i] = 0
                return

    def doesMoveEndGame(self, col, row, turn):	
        ''' 
            Returns whether moving in a particular position ends the game \n
            Returns 1, 0, or -1 if this particular move ends the game, otherwise return None
        '''

        # Check whether piece creates vertical 4 in a row
        count = 1
        for rowBelow in range(row-1, max(row-4,-1), -1):
            if self.board[col][rowBelow] == turn:
                count += 1
                if count == 4:
                    return turn
        
        
        # Check if piece creates horizontal 4 in a row
        count = 1

        for colRight in range(col+1, min(col+4,7)):
            if self.board[colRight][row] == turn:
                count += 1
            else:
                break	

        for colLeft in range(col-1, max(col-4,-1), -1):
            if self.board[colLeft][row] == turn:
                count += 1
            else:
                break

        if count >= 4:
            return turn


        # Check whether piece create diagonal TL to BR
        count = 1

        for offset in range(1, 4):		# Bottom right of piece
            if col+offset==7 or row-offset==-1 or self.board[col+offset][row-offset] != turn:
                break
            count += 1

        for offset in range(1, 4):	# Top left of piece
            if col-offset==-1 or row+offset==6 or self.board[col-offset][row+offset] != turn:
                break
            count += 1

        if count >= 4:
            return turn


        # Check whether piece create diagonal TR to BL
        count = 1

        for offset in range(1, 4):		# Bottom left of piece
            if col-offset==-1 or row-offset==-1 or self.board[col-offset][row-offset] != turn:
                break
            count += 1

        for offset in range(1, 4):	# Top right of piece
            if col+offset==7 or row+offset==6 or self.board[col+offset][row+offset] != turn:
                break
            count += 1

        if count >= 4:
            return turn

        # Game is tie since all squares are filled
        if self.turnNum == 42:
            return 0

        # Move does NOT end game
        return None

    def isValidMove(self, col):		# Returns whether a move is valid
        return self.board[col][5] == 0

    def getValidMove(self, col):
        ''' 
            Returns the y index of a move in a particular column \n
            If no move valid, returns -1
        '''
        for row in range(6):
            if self.board[col][row] == 0:
                return row
        return -1

    def resetGame(self):
        self.board.fill(0)
        self.turn = 1
        self.turnNum = 0

    def getBoardState(self):
        ''' Returns a flattened array of board for current player '''
        return self.board.flatten() * self.turn

    def getMove(self, model, exploit=1):
        '''
            Calculates the move made by an agent in the current game. \n
            Returns various values such as the move, its Q-value, and the state input in a dictionary. \n
        '''
        state = self.getBoardState()
        output = model.predict(np.array([state]))[0]
        
        movePredictions = output[:7]
        movePredictions *= self.getValidMoves() # Set probability of all non-valid moves to zero
        move = self.selectMove(movePredictions, exploit)
        # move = self.selectMoveProb(movePredictions)
        qValue = movePredictions[move]
        
        return {
            "state": state,
            "move": move,
            "qValue": qValue,
            "movePredictions": movePredictions
        }

    def selectMove(self, predictions, exploit=1):
        ''' 
            Selects index of highest q-value from predictions array with probability equal to exploit
            Otherwise returns a random (valid) move
        '''
        if exploit == 1 or random.random() <= exploit:
            return predictions.argmax()

        return random.sample([i for i in range(7) if predictions[i] != 0], 1)[0]

    def selectMoveProb(self, predictions):
        prob = predictions / np.sum(predictions)
        return np.random.choice(list(range(7)), p=prob)

    def getValidMoves(self):
        '''
            Returns an array of valid moves
            Elements are True if valid, and False if not
        '''
        return np.array([self.isValidMove(i) for i in range(7)])



    def getMaxQValue(self, model, sPrime):
        '''
            Returns the maximum Q-value for an sPrime
            If sPrime is 0 or 1, it is a winning state and will return that number
            Assuming sPrime is not a winning state, it will be an array of valid sa inputs from a state 
        '''
        if type(sPrime) == int:
            return sPrime

        predictions = model.predict(np.array([sPrime])) # Predict outputs for each sPrime
        return max(predictions[0])


class RandomAgent():
    def __init__(self, game):
        self.game = game

    def getMove(self):
        validMoves = []

        for i in range(7):		# for all possible moves
            if self.game.isValidMove(i):
                validMoves.append(i)

        return random.sample(validMoves, 1)[0]

    def playMove(self):
        return self.game.move(self.getMove())



class GoodAgent():
    def __init__(self, game, winningPlacements=True):
        self.game = game
        self.validMoves = np.zeros(7, dtype="int8")
        self.validMoveCount = 0

        # Whether or not agent will always place in winning placements
        self.winningPlacements = winningPlacements

    def playMove(self):
        return self.game.move(self.getMove())

    def getMove(self):

        # Check for winning placements
        if self.winningPlacements:
            for col in range(7):
                # If move has no free slots, skip this column
                if self.game.board[col][5] != 0:
                    continue

                # Find row which piece would be placed in this column
                row = 0
                for row in range(6):
                    if self.game.board[col][row] == 0:
                        break

                winner = self.game.doesMoveEndGame(col, row, self.game.turn)

                if winner == None:
                    continue
                return col


        # Return a random 'safe' move
        self.validMoves.fill(0)
        self.validMoveCount = 0

        
        # Check for blocking placements
        for col in range(7):
            # If move has no free slots, skip this column
            if self.game.board[col][5] != 0:
                continue

            # Find row which piece would be placed in this column
            row = 0
            for row in range(6):
                if self.game.board[col][row] == 0:
                    break


            # If piece blocks opponent, always play here
            winner = self.game.doesMoveEndGame(col, row, -self.game.turn)
            if winner == -self.game.turn:
                return col


            # Add moves that don't cause opponent to win next turn (safe moves)
            if row == 5 or self.game.doesMoveEndGame(col, row+1, -self.game.turn) == None:
                self.validMoves[self.validMoveCount] = col
                self.validMoveCount += 1


        # Pick a random move from this list of 'safe' moves
        if self.validMoveCount > 0:
            #return self.validMoves[ self.game.turnNum % self.validMoveCount ]
            return self.validMoves[ random.randint(0,self.validMoveCount-1) ]


        # Return a random move
        self.validMoves.fill(0)
        self.validMoveCount = 0
        for i in range(7):		# Add all possible moves to valid moves list
            if self.game.isValidMove(i):
                self.validMoves[self.validMoveCount] = i
                self.validMoveCount += 1
        #return self.validMoves[ self.game.turnNum % self.validMoveCount ]
        return self.validMoves[ random.randint(0,self.validMoveCount-1) ]




# class botNN():
# 	def __init__(self, game, NN):
# 		self.game = game
# 		self.NN = NN

# 		# Used for bot to loop through each future move and give it a value
# 		self.moveRanks = np.array([-1.]*7)

# 		self.selectionNoise = False

# 	def getMove(self):

# 		for i in range(7):	# for all possible moves
# 			if self.game.isValidMove(i):		# if move is valid
# 				# gets bots rating of this move

# 				# get input for NN
# 				boardInput = self.game.getBotInput()

# 				# make future move in this board input
# 				replacementIndex = i*6
# 				while boardInput[replacementIndex] != 0:
# 					replacementIndex += 1
# 				boardInput[replacementIndex] = 1

# 				self.moveRanks[i] = self.NN.getOutput(boardInput)[0]

# 		selectedMove = -1
# 		if self.selectionNoise:
# 			# Select highest rated move with certain probability
# 			while True:
# 				selectedMove = self.moveRanks.argmax()	# get highest rated move
# 				if random.random() < 0.90 or self.moveRanks.max() == -1:	# pick move with 90% probability (or if all moves have been selected, select this move)
# 					break
# 				self.moveRanks[selectedMove] = -1
# 		else:
# 			selectedMove = self.moveRanks.argmax()	# get highest rated move
        
# 		self.moveRanks.fill(-1)		# reset move ranks

# 		return selectedMove
