import os, sys, random, math
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

# Load or create new model
if len(sys.argv) == 2:
    model = keras.models.load_model("models/" + sys.argv[1])
else:
    model = createModel()
    print("No model entered, testing new model.")
model.summary()

# Create base game
game = Connect4Game()
opponent = GoodAgent(game, True)
# opponent = RandomAgent(game)


def resultsString(winners):
    '''
        Returns a string showing the stats of wins/draws/losses from an array of winners
    '''
    return ("Won: {}/{n} - Tie: {}/{n} - Loss: {}/{n}".format(
        winners.count(1), 
        winners.count(0),
        winners.count(-1),
        n = len(winners)
    ))



game_count = int(input("Number of games? "))

# Play games and store results
gameResults = playTrainingGames(
    game_count, 
    game, 
    model, 
    opponent
)

# Print results
print(resultsString(gameResults["winners"]))

print("Done")

