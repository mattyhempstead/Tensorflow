import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# print("tf version", tf.__version__)

# Stop some of the random logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def getModel():
    if len(sys.argv) == 2:
        model = keras.models.load_model("models/" + sys.argv[1])
        return model
    else:
        model = keras.Sequential()

        # model.add(keras.layers.Conv2D(128, (2, 2), activation='tanh', input_shape=(7, 6, 1)))
        # model.add(keras.layers.Conv2D(128, (2, 2), activation='tanh'))
        # model.add(keras.layers.Conv2D(256, (2, 2), activation='tanh'))
        # model.add(keras.layers.Conv2D(256, (2, 2), activation='tanh'))

        model.add(keras.layers.Conv2D(128, (3, 3), activation='tanh', input_shape=(7, 6, 1)))
        model.add(keras.layers.Conv2D(256, (3, 3), activation='tanh'))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation='tanh'))
        model.add(keras.layers.Dense(256, activation='tanh'))
        model.add(keras.layers.Dense(256, activation='tanh'))
        # model.add(keras.layers.Dense(256, activation='tanh'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        # model.add(keras.layers.Dropout(0.1))

        model.compile(
            optimizer='adam', #tf.keras.optimizers.Adam(0.001)
            loss=keras.losses.binary_crossentropy,
            metrics=[],
        )
        return model



model = getModel()
model.summary()



from connect4 import Connect4Game, RandomAgent, GoodAgent
from playTrainingGames import playTrainingGames

game = Connect4Game()
# opponent = RandomAgent(game)
opponent = GoodAgent(game)


def resultsString(winners):
    '''
        Returns a string showing the stats of wins/draws/losses from an array of winners
    '''
    return ("Won: {}/{n}, Tie: {}/{n}, Loss:{}/{n}".format(
        winners.count(1), 
        winners.count(0),
        winners.count(-1),
        n = len(winners)
    ))

def trainModel(model, states, qValues):
    # Now train on q values and board states
    model.fit(
        states, 
        qValues,
        epochs = 1,
        batch_size = 256,
        # validation_data = (testImages, testLabels),
    )



GAME_COUNT = 300
TRAIN_COUNT = 0

qValues = []
states = []

results = {
    "wins": []
}

# Get initial stats on model
gameResults = playTrainingGames(GAME_COUNT, game, model, opponent)
qValues = np.array(list(qValues) + list(gameResults["pastBoardQValues"]))
states = np.array(list(states) + list(gameResults["pastBoardStates"]))
print("Initial Score - {}".format(resultsString(gameResults["winners"])))

while True:
    count = int(input("Number of times? "))
    if count == 0: break

    for i in range(count):

        trainModel(model, states, qValues)

        gameResults = playTrainingGames(GAME_COUNT, game, model, opponent)

        qValues = np.array(list(qValues) + list(gameResults["pastBoardQValues"]))
        states = np.array(list(states) + list(gameResults["pastBoardStates"]))
        # qValues = gameResults["pastBoardQValues"]
        # states = gameResults["pastBoardStates"]

        print("After Train - {}".format(resultsString(gameResults["winners"])))

        if not os.path.exists("models/cp"):	
            os.mkdir("models/cp")
        model.save("models/cp/model_{}-{}.h5".format(
            len(os.listdir("models/cp")),
            gameResults["winners"].count(1)
        ))

        # Add wins result
        results["wins"].append(gameResults["winners"].count(1))



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

gameResults = playTrainingGames(100, game, model, opponent)
print(resultsString(gameResults["winners"]))


if input("Save model? ").upper() == "Y":
    model.save("models/model.h5")
    print("Saved model as model.h5")

print("Done")


