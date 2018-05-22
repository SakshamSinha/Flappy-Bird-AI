# Author Corentin Chéron
# from random agent at
# https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/tests/test_ple.py

import numpy as np
from ple import PLE
import random
import os, sys
import pygame
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FlappyBirdClone import flappy
from joblib import dump, load
import datetime, time
from shutil import copyfile
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Add
from keras.layers.merge import Multiply
from keras.optimizers import SGD, RMSprop



class QAgent():
    """An approximate q-learning agent inspired from the Pacman project."""
    jFilePath = "q-agentNN-stats.jldump"
    wFilePath = "q-agentNN-weights.jldump"
    def __init__(self, actions, stateSize,
                 discount=0.99, learningRate=.005, epsilon=.8):
        # Takes a list of feature function as input
        self.possibleActions = actions
        self.possibleActionsInt = list(range(len(self.possibleActions)))

        self.discount = discount
        self.learningRate = learningRate
        self.previousstate = None
        self.previousAction = None
        self.history = []
        self.epsilon = epsilon
        self.steps = 0
        self.episodes = 0
        self.stats = []
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=stateSize, activation='relu'))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(len(self.possibleActions)))

        optimizer = RMSprop(lr=self.learningRate, rho=0.9, epsilon=0.0001)
        self.loadQ()
        self.model.compile(loss='mse', optimizer=optimizer)

# Compile model

    def loadQ(self):
        if os.path.exists(self.jFilePath):
            self.steps, self.episodes, self.stats = load(self.jFilePath)
            print("loaded Q-Values from previous steps={}".format(
                self.steps))
        if os.path.exists(self.wFilePath):
            self.model.load_weights(self.wFilePath)
            print("Loaded weights to NN")


    def dumpQ(self):
        self.model.save_weights(self.wFilePath)
        dump((self.steps, self.episodes, self.stats), self.jFilePath)
        print("weights and stats saved")

    def addStat(self, tup):
        self.stats.append(tup)

    def updateEpsilon(self, epsilon):
        self.epsilon = epsilon

    def getQvalues(self, state):
        qv = self.model.predict(state.reshape(-1, 8))[0]
        return qv

    def getQvalue(self, state, action):
        qv = self.getQvalues(state)[action]
        return qv

    def randomAction(self):
        val = random.choice(self.possibleActionsInt + [1] * 16)
        return val

    def getActionAndSaveState(self, state):
        if random.random() < self.epsilon:
            self.previousAction = self.randomAction()
        else:
            qa, qb = self.getQvalues(state)
            # print("chossing between {} {}".format(qa, qb))
            if qa == qb:
                # print("equal thus random")
                self.previousAction = self.randomAction()
            elif qa > qb:
                self.previousAction = 0
            else:
                self.previousAction = 1
        self.previousstate = state
        return self.possibleActions[self.previousAction]

    def getValue(self, state):
        return max(self.getQvalues(state))

    def updateQ(self, prevstate, prevAction, state, reward, terminal):
        prevQ = self.getQvalues(prevstate)
        if not terminal:
            target = reward + self.discount * self.getValue(state)
        else:
            target = reward
        prevQ[prevAction] = target
        self.model.train_on_batch(np.array([prevstate]), prevQ.reshape(-1, 2))

        diff = (target - prevQ[prevAction])
        return diff

    def updateAndGetAction(self, reward, state, terminal):
        self.steps += 1
        if self.steps % 10000 == 0:
            self.dumpQ()

        if self.previousstate is None:
            return 0., self.getActionAndSaveState(state)
        # print("newstate")
        self.history.append((self.previousstate, self.previousAction,
                             state, reward, terminal))
        loss = self.updateQ(self.previousstate, self.previousAction, state, reward,
                            terminal)
        return loss, self.getActionAndSaveState(state)


    def updateHistory(self):
        """backward experience learning..."""
        # print("updating history of len={}".format(len(self.history)))
        for prevstate, prevA, state, reward, terminal in reversed(self.history):
            self.updateQ(prevstate, prevA, state, reward, terminal)
            self.history = []


def preprocessor(state):
    return np.array([
        state[k] for k in sorted(state.keys())
    ])

# order of data in state
#[0'next_next_pipe_bottom_y',
# 1'next_next_pipe_dist_to_player',
# 2'next_next_pipe_top_y',
# 3'next_pipe_bottom_y',
# 4'next_pipe_dist_to_player',
# 5'next_pipe_top_y',
# 6'player_vel',
# 7'player_y']
###################################
def testAgent(qAgent):
    qAgent.updateEpsilon(0.)
    avgScore = 0.
    scoreCnt = 0
    numGame = 50
    maxScore = -5
    env.reset_game()
    print("Test {} games".format(numGame))
    while numGame:
        if env.game_over(): # if the game is over, reset
            avgScore += env.game.getScore()
            maxScore = max(maxScore, env.game.getScore())
            scoreCnt += 1
            numGame -= 1
            env.reset_game()

        reward = env.act(qAgent.getActionAndSaveState(env.getGameState()))

    avgScore /= scoreCnt
    print("Average score: {}".format(avgScore))
    return avgScore, maxScore


if len(sys.argv) < 2 :
    print("Error argument: please specify <test_name>")
    sys.exit(-1)

# handle timestamp folder
timestr = datetime.datetime.fromtimestamp(
    time.time()).strftime('%Y-%m-%d_%H:%M:%S')
folder = timestr + "_" + sys.argv[1]

if os.path.exists(folder):
    print("Error {} already exists".format(folder))
    sys.exit(-1)
os.mkdir(folder)
copyfile(__file__, os.path.join(folder, __file__))

FPS = 60

STEPS_PER_EPOCHS = 2000
EPOCHS = 2000
EPSILON_START = .1
EPSILON_DECAY = EPOCHS * STEPS_PER_EPOCHS
EPSILON_MIN = 0.01
EPSILON_DECAY_V = (EPSILON_MIN - EPSILON_START) / EPSILON_DECAY
SEED = 123456
epsilon = EPSILON_START
rng = np.random.RandomState(SEED)
game = flappy.FlappyClone()
env = PLE(game, display_screen=True, force_fps=True, fps=30,
          state_preprocessor=preprocessor, rng=rng)
env.game.rewards["positive"] = 1
# env.game.rewards["tick"] = .01
qAgent = QAgent(env.getActionSet(), env.getGameStateDims()[0], discount=.99,
                       learningRate=.0025,  epsilon=epsilon)
qAgent.jFilePath = os.path.join(folder, qAgent.jFilePath)

reward = 0.
clock = pygame.time.Clock()
laststate = None
lastticks = 0
periodJump = 0
action = None
nextTest = False
for e in range(EPOCHS):
    avgloss = 0.
    avgreward =  0.
    maxScore = -5.
    avgAction = 0.

    for s in range(STEPS_PER_EPOCHS):
        if env.game_over(): # if the game is over, reset
            # print("tick {} death at score: {}".format(e * STEPS_PER_EPOCHS + s, game.getScore()))
            qAgent.episodes += 1
            qAgent.updateHistory()
            if nextTest:
                avgScore, maxScore = testAgent(qAgent)
                qAgent.addStat((qAgent.steps, qAgent.episodes, (maxScore, avgScore)))
                qAgent.updateEpsilon(epsilon)
                nextTest = False

            qAgent.previousstate = None
            # env.game.rng.seed(SEED)
            env.reset_game()
            periodJump = 0

        # periodJump += 1
        # if periodJump%18==0:
            # reward = env.act(pygame.K_UP)
        # else:
            # reward = env.act(None)

        reward = env.act(action)
        avgAction += action == pygame.K_UP
        state = env.getGameState()
        loss, action = qAgent.updateAndGetAction(reward, state, env.game_over())

        avgloss += loss
        avgreward += reward

        # update epsilon
        epsilon += EPSILON_DECAY_V
        if epsilon < EPSILON_MIN:
            epsilon = EPSILON_MIN
        qAgent.updateEpsilon(epsilon)
        # clock.tick_busy_loop(FPS)

    if e % 10 == 0:
        nextTest = True

    print("Epoch {}: Steps:{} episodes:{} epsilon={:5.3f}, avgLoss={},"
          "totalReward={:5.3f}, maxScore={}".format(
        e, qAgent.steps, qAgent.episodes,
        epsilon, avgloss / STEPS_PER_EPOCHS, avgreward, maxScore))
    print("avg Action {}".format(avgAction / STEPS_PER_EPOCHS))



