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

# Load or create new model
if len(sys.argv) == 2:
    model = keras.models.load_model("models/" + sys.argv[1])
else:
    model = createModel()
model.summary()

# Create base game
game = Connect4Game()
opponent = GoodAgent(game)
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

def trainModel(model, states, rewards):
    # Now train on q values and board states
    model.fit(
        states, 
        rewards,
        epochs = 1,
        batch_size = 256,
        # validation_data = (testImages, testLabels),
    )



GAME_COUNT = 1000

rewards = np.array([])
states = np.array([])

results = {
    "wins": []
}

# Get initial stats on model
gameResults = playTrainingGames(GAME_COUNT, game, model, opponent)
rewards = np.array(list(rewards) + list(gameResults["pastBoardRewards"]))
states = np.array(list(states) + list(gameResults["pastBoardStates"]))
results["wins"].append(gameResults["winners"].count(1))
print("Initial Score - {}".format(resultsString(gameResults["winners"])))

while True:
    count = int(input("Number of times? "))
    if count == 0: break

    for i in range(count):

        trainModel(model, states, rewards)

        gameResults = playTrainingGames(GAME_COUNT, game, model, opponent)

        selectedTrains = np.array([random.random()<0.9 for i in range(len(rewards))])
        rewards = np.array(list(rewards[selectedTrains]) + list(gameResults["pastBoardRewards"]))
        states = np.array(list(states[selectedTrains]) + list(gameResults["pastBoardStates"]))

        # Using this with GAME_COUNT=100 give mostly stable training
        # rewards = np.array(list(rewards[::2]) + list(gameResults["pastBoardRewards"]))
        # states = np.array(list(states[::2]) + list(gameResults["pastBoardStates"]))

        # rewards = np.array(list(rewards) + list(gameResults["pastBoardRewards"]))
        # states = np.array(list(states) + list(gameResults["pastBoardStates"]))
        # rewards = gameResults["pastBoardRewards"]
        # states = gameResults["pastBoardStates"]

        print("Train: {}/{} - {}".format(
            i+1,
            count,
            resultsString(gameResults["winners"])
        ))

        # Add wins result
        results["wins"].append(gameResults["winners"].count(1))

        # Save model
        if not os.path.exists("models/cp"):	
            os.mkdir("models/cp")
        # Only save model if score is at least 90% the score of best model
        if results["wins"][-1] >= max(results["wins"])*0.9:     
            model.save("models/cp/model_{}-{}.h5".format(
                len(os.listdir("models/cp")),
                gameResults["winners"].count(1)
            ))


    # Graph results so far
    plt.plot(
        range(len(results["wins"])),
        results["wins"],
        label='Wins'
    )
    # plt.xticks(range(epochs))
    plt.legend()
    plt.show()




print("Finished Training, Evaluating Final Agent...")

gameResults = playTrainingGames(GAME_COUNT, game, model, opponent)
print(resultsString(gameResults["winners"]))


if input("Save model? ").upper() == "Y":
    model.save("models/model.h5")
    print("Saved model as model.h5")

print("Done")


