
import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# print("tf version", tf.__version__)

# Seed setting doesn't seem to work at all :(
# os.environ['PYTHONHASHSEED'] = '0'
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0);
# tf.compat.v1.random.set_random_seed(0)

# Stop some of the random logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



model = keras.models.load_model("models/model.h5")
# model.summary()


from connect4 import Connect4Game, RandomAgent, GoodAgent
from playTrainingGames import playTrainingGames, getMoveRanks

game = Connect4Game()


while True:
    if game.turn == 1:
        moveRanks = getMoveRanks(game, model)
        move = moveRanks.argmax()
        winner = game.move(move)

    else:
        game.printGame()
        
        while True:
            move = int(input("Move: "))-1
            if 0 <= move <= 6 and game.isValidMove(move):
                break
            print("Invalid Move")
            
        winner = game.move(move)

    if winner != None:
        game.printGame()
        print("Winner: {}".format(winner))
        break



