# Author Corentin Chéron
# from random agent at
# https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/tests/test_ple.py

import numpy as np
from ple import PLE
import os, sys
import pygame
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FlappyBirdClone import flappy


class ApproxQAgent():
    """An approximate q-learning agent inspired from the Pacman project."""
    def __init__(self, actions, obsSize, featuresFuns,
                 discount=0.99, learningRate=.005):
        # Takes a list of feature function as input
        self.possibleActions = actions
        self.possibleActionsInt = list(range(len(self.possibleActions)))
        self.w = np.random.rand(len(featuresFuns))
        # self.newW = self.w.copy()
        self.featuresFuns = featuresFuns
        self.discount = discount
        self.learningRate = learningRate
        self.previousObs = np.zeros((obsSize), dtype=np.float32)
        self.previousAction = 0

    def getQvalue(self, state, action):
        return sum([f(state, action) * self.w[i]
                    for i, f in enumerate(self.featuresFuns)])

    def getAction(self, state, epsilon):
        if np.random.rand() < epsilon:
            self.previousAction = np.random.choice(self.possibleActionsInt)
        else:
            self.previousAction = max(
                [(self.getQvalue(state, action), action) for action
                 in self.possibleActionsInt], key=lambda x: x[0])[1]
        # print("." if self.previousAction else "*", end="")
        return self.possibleActions[self.previousAction]

    def getValue(self, state):
        return max([self.getQvalue(state, action) for action
                    in self.possibleActionsInt])

    def updateWeights(self, obs, reward):
        sumDiff = 0.
        for i, f in enumerate(self.featuresFuns):
            value = f(self.previousObs, self.previousAction)
            diff = (reward + self.discount * self.getValue(obs)
                    - self.getQvalue(self.previousObs, self.previousAction))
            # print("feat{}, diff={}, value={}".format(i, diff, value))
            self.w[i] += self.learningRate * diff * value
            sumDiff += diff
        return sumDiff

    def update(self, reward, obs):
        # print("obs=", obs)
        loss = self.updateWeights(obs, reward)
        self.previousObs = obs
        return loss

class AutoScaler(object):
    def __init__(self):
        self.min = None
        self.max = None

    def normalise(self, value):
        self.min = min(self.min, value) if self.min else value
        self.max = max(self.max, value) if self.max else value
        if self.max == self.min:
            return 1.
        return (value - self.min) / (self.max - self.min)

scalers = [AutoScaler() for idx in range(8)]
def preprocessor(state):
    return np.array([scalers[i].normalise(state[k]) for i, k in
                     enumerate(sorted(state.keys()))])

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

features = [
    lambda x, y: 1.,
    lambda x, y: (x[3] - x[7]) * y , # botpipe - playery
    lambda x, y: (x[3] - x[7]) * (1 - y) ,
    lambda x, y: (x[5] - x[7]) * y , # toppipe - playery
    lambda x, y: (x[5] - x[7]) * (1 - y) , # bot - playery
    lambda x, y: x[6] * y, # velocity
    lambda x, y: x[6] * (1 - y),
    lambda x, y: x[4] * y, #  pipe distance
    lambda x, y: x[4] * (1 - y),
    # lambda x, y: (x[3] - x[7]) / (x[6] ) * (1 - y) + ,
]


STEPS_PER_EPOCHS = 1000
EPOCHS = 60
EPSILON_START = 0.01
EPSILON_DECAY = EPOCHS * STEPS_PER_EPOCHS
EPSILON_MIN = 0.00000
EPSILON_DECAY_V = (EPSILON_MIN - EPSILON_START) / EPSILON_DECAY

game = flappy.FlappyClone()
env = PLE(game, display_screen=True, force_fps=True, fps=30,
          state_preprocessor=preprocessor)
env.init()
approxQAgent = ApproxQAgent(env.getActionSet(), env.getGameStateDims(),
                      features, learningRate=.002)

reward = 0.
epsilon = EPSILON_START
for e in range(EPOCHS):
    avgloss = 0.
    avgreward =  0.
    for s in range(STEPS_PER_EPOCHS):
        if env.game_over(): # if the game is over, reset
            # print("tick {} death at score: {}".format(e * STEPS_PER_EPOCHS + s, game.getScore()))
            env.reset_game()
            obs = env.getGameState()
            action = approxQAgent.getAction(obs, epsilon)

        obs = env.getGameState()
        loss = approxQAgent.update(reward, obs)
        action = approxQAgent.getAction(obs, epsilon)
        reward = env.act(action)
        avgloss += loss
        avgreward += reward

        # update epsilon
        epsilon += EPSILON_DECAY_V
        if epsilon < EPSILON_MIN:
            epsilon = EPSILON_MIN


    print("Epoch {}: epsilon={:5.3f}, avgLoss={:5.3f}, avgReward={:5.3f} ".format(
        e, epsilon, avgloss / STEPS_PER_EPOCHS, avgreward / STEPS_PER_EPOCHS ))
print("weights=", approxQAgent.w)

for s in scalers:
    print("scaler: min:{} max:{}".format(s.min, s.max))
print("Test...")
for e in range(10):
    while True:
        if env.game_over(): # if the game is over, reset
            print("test {}, death at score: {}".format(e, game.getScore()))
            env.reset_game()
            break
        reward = env.act(approxQAgent.getAction(env.getGameState(), 0.))

