

# for fileName in cp:
#     x = fileName.split("-")[1].split(".")[0]
#     x = int(x)
#     if x < 87:
#         # delete file
#         os.remove("cp/" + fileName)

# def s(fileName):
#     return int(fileName.split("-")[1].split(".")[0])

# cp.sort(key=s, reverse=True)

# print(cp)

# for i,fileName in enumerate(cp):
#     print("cp/" + str(i) + "_" + fileName.split("-")[1].split(".")[0] + ".h5")
#     os.rename(
#         "cp/" + fileName, 
#         "cp/" + str(i) + "_" + (fileName.split("-")[1].split(".")[0]) + ".h5"
#     )

# # os.rename("cp/" + cp[0], "cp/" + cp[0] + "a")



import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from connect4 import Connect4Game, RandomAgent, GoodAgent
from playTrainingGames import playTrainingGames
from createModel import createModel

# Stop some of the random logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Create base game
game = Connect4Game()
opponent = GoodAgent(game)
# opponent = RandomAgent(game)


GAME_COUNT = 10000


cp = os.listdir("models/cp")

for fileName in cp:

    print(fileName)

    oldScore = float(fileName.split("_")[1][:-3])
    

    model = keras.models.load_model("models/cp/" + fileName)

    gameResults = playTrainingGames(GAME_COUNT, game, model, opponent)

    wins = gameResults["winners"].count(1)
    wins /= 30

    newScore = round((wins + oldScore)/2, 2)

    newName = "models/cp/" + fileName.split("_")[0] + "_" + str(newScore) + ".h5"

    print(wins, newName)

    os.rename(
            "models/cp/" + fileName, 
            newName
    )

