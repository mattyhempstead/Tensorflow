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


def trainModel(model, saPairs, qValues):
    # Now train on q values and state-action pairs
    model.fit(
        saPairs, 
        qValues,
        epochs = 1,
        batch_size = 256,
        # validation_data = (testImages, testLabels),
    )

def saveModel(model, wins):
    if not os.path.exists("models/cp"):	
        os.mkdir("models/cp")

    # Only save model if score is at least x% the score of best model
    if results["wins"][-1] >= max(results["wins"])*0.9:     
        model.save("models/cp/model_{}_{}.h5".format(
            len(os.listdir("models/cp")),
            wins
        ))


GAME_COUNT = 100
MEMORY_RECALL_SIZE = 500
MEMORY_MAX_SIZE = 50000     # Approximately 10 samples per game (~1GB per 100000?)

# Total number of times run train operation
trainCount = 0

# Memory for previous states
# Might also need to store whether game was won or lost at one point
memory = []

results = {
    "wins": []
}


while True:
    count = int(input("Number of times? "))
    if count == 0: break

    for i in range(count):

        # Play games and store results
        gameResults = playTrainingGames(
            GAME_COUNT, 
            game, 
            model, 
            opponent,
            discountFactor = 0.9, 
            exploit = trainCount / (trainCount + 20) #1 - math.exp(-trainCount/100)
        )

        # Print results
        print("#{}/{} - {}".format(
            i,
            count,
            resultsString(gameResults["winners"])
        ))

        # Add wins result
        results["wins"].append(gameResults["winners"].count(1))


        # Save model if not initial train 
        if i != 0:
            saveModel(model, gameResults["winners"].count(1))


        # Train model

        
        # Get some s-a-s' triplets from memory
        print("Gathering memory states...")
        memoryStates = random.sample(memory, min(MEMORY_RECALL_SIZE, len(memory)))
        saPairs = [memory[0] for memory in memoryStates]
        sPrimes = [memory[1] for memory in memoryStates]

        # Calculate the max q-value for each s'
        print("Predicting Q-values for memory states...")
        qValues = [[game.getMaxQValue(model, sPrime)] for sPrime in sPrimes]


        # print(np.array([[game.getMaxQValue(model, sPrime)] for sPrime in sPrimes]).shape)

        # qValues = np.concatenate((qValues, [[game.getMaxQValue(model, sPrime)] for sPrime in sPrimes]))

        # print(len(qValues), len(sPrimes), len(saPairs))
        # for i in range(len(saPairs)):
        #     print(qValues[i], saPairs[i])

        # print(qValues, gameResults["qValues"])

        # Add new experience to current training data 
        print("Adding new experience to current training states...")
        saPairs += list(gameResults["saPairs"])
        qValues += list(gameResults["qValues"])
        
        saPairs = np.array(saPairs)
        qValues = np.array(qValues)

        # Train model with the selected states
        trainModel(model, saPairs, qValues)


        # Add new experience to memory
        # Add a maximum of MEMORY_MAX_SIZE memories before replacing old ones
        #memory += list(zip(gameResults["saPairs"], gameResults["sPrimes"]))
        for newMemory in zip(gameResults["saPairs"], gameResults["sPrimes"]):
            if len(memory) < MEMORY_MAX_SIZE:
                memory.append(newMemory)
            else:
                memory[random.randrange(MEMORY_MAX_SIZE)] = newMemory
        print("Memory size: {}".format(len(memory)))




        # selectedTrains = np.array([random.random()<0.9 for i in range(len(rewards))])
        # rewards = np.array(list(rewards[selectedTrains]) + list(gameResults["pastBoardRewards"]))
        # states = np.array(list(states[selectedTrains]) + list(gameResults["pastBoardStates"]))


        # Using this with GAME_COUNT=100 give mostly stable training
        # rewards = np.array(list(rewards[::2]) + list(gameResults["pastBoardRewards"]))
        # states = np.array(list(states[::2]) + list(gameResults["pastBoardStates"]))

        # rewards = np.array(list(rewards) + list(gameResults["pastBoardRewards"]))
        # states = np.array(list(states) + list(gameResults["pastBoardStates"]))
        # rewards = gameResults["pastBoardRewards"]
        # states = gameResults["pastBoardStates"]

        
        trainCount += 1


    # Print final performance after all training epochs
    gameResults = playTrainingGames(GAME_COUNT, game, model, opponent)
    print("Final - {}".format(resultsString(gameResults["winners"])))
    # Save model
    saveModel(model, gameResults["winners"].count(1))

    

    # Graph results so far
    plt.plot(
        range(len(results["wins"])),
        results["wins"],
        label='Wins'
    )
    # plt.xticks(range(epochs))
    plt.legend()
    plt.show()




print("Done")

