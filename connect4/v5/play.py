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
randomAgent = RandomAgent(game)

while True:
    if game.turn == 1:
        move = game.getMove(model, 1)
        winner = game.move(move["move"])

        # # Print predictions from agent
        # winPredictions = [" {:2d}%".format(int(100*i)) for i in move["winMoves"]]
        # winPredictions = [
        #     winPredictions[move["validMoves"].index(i)] 
        #     if i in move["validMoves"] else "  0%" 
        #     for i in range(7)
        # ]
        
        movePredictions = ["{:.2f}".format(i) for i in move["movePredictions"]]
        # predictions = [
        #     predictions[move["validMoves"].index(i)] 
        #     if i in move["validMoves"] else "0.00" 
        #     for i in range(7)
        # ]

        print("")
        print("Agent played move {}".format(move["move"]+1))
        print("  ".join(movePredictions))
        # print("  ".join(winPredictions))

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
                print("GoodAgent played move {}".format(move+1))
                break
            elif move.upper() == "R":
                move = randomAgent.getMove()
                print("RandomAgent played move {}".format(move+1))
                break
                    
            print("Invalid Move")
            
        winner = game.move(move)

    if winner != None:
        print(game)
        print("Winner: {}".format(winner))
        break


