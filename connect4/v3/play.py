import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from connect4 import Connect4Game, RandomAgent, GoodAgent


# Stop some of the random logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    model = keras.models.load_model("models/" + sys.argv[1])
except:
    print("Failed to load model")
    sys.exit()


game = Connect4Game()
goodAgent = GoodAgent(game)

while True:
    if game.turn == 1:
        move = game.getMove(model)
        winner = game.move(move["move"])

        # Print predictions from agent
        predictions = [str(round(i[0], 2)) for i in move["predictions"]]
        predictions = [
            predictions[move["validMoves"].index(i)] 
            if i in move["validMoves"] else "0" 
            for i in range(7)
        ]
        print("")
        print(", ".join(predictions))

    else:
        print(game)
        
        while True:
            move = input("Move: ")
            if move.isdigit():
                move = int(move)-1
                if 0 <= move <= 6 and game.isValidMove(move):
                    break
            elif move.upper() == "G":
                move = goodAgent.getMove()
                break
                    
            print("Invalid Move")
            
        winner = game.move(move)

    if winner != None:
        print(game)
        print("Winner: {}".format(winner))
        break


