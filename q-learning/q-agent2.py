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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from joblib import dump, load
import datetime, time
from shutil import copyfile


class QAgent():
    """An approximate q-learning agent inspired from the Pacman project."""
    jFilePath = "q-agent2-qvalues.jldump"
    def __init__(self, actions, stateSize, gridSize=10,
                 discount=0.99, learningRate=.005, epsilon=.8):
        # Takes a list of feature function as input
        self.gridSize = gridSize
        self.possibleActions = [None, pygame.K_UP]
        self.possibleActionsInt = list(range(len(self.possibleActions)))
        matrixSize = stateSize + [len(self.possibleActions)]

        self.q = np.zeros(tuple(matrixSize))
        self.maxq = 1.
        print("initializing matrix of size {}={}".format(matrixSize,
                                                         self.q.size))
        self.discount = discount
        self.learningRate = learningRate
        self.previousstate = None
        self.previousAction = None
        self.history = []
        self.epsilon = epsilon
        self.loadQ()
        self.steps = 0
        self.episodes = 0
        self.stats = []

    def loadQ(self):
        if os.path.exists(self.jFilePath):
            self.q, self.steps, self.episodes, self.stats = load(self.jFilePath)
            print("loaded Q-Values from previous steps={}".format(
                self.steps))


    def dumpQ(self):
        dump((self.q, self.steps, self.episodes, self.stats), self.jFilePath)
        print("Q-Values saved")

    def addStat(self, tup):
        self.stats.append(tup)

    def updateEpsilon(self, epsilon):
        self.epsilon = epsilon

    def getQvalue(self, state, action):
        return self.q[state][action]

    def randomAction(self):
        val = random.choice(self.possibleActionsInt + [0] * 16)
        # print("random", val)
        return val

    def getActionAndSaveState(self, state):
        if random.random() < self.epsilon:
            self.previousAction = self.randomAction()
        else:
            qa, qb = self.q[state]
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
        return self.q[state].max()

    def updateQ(self, prevstate, prevAction, state, reward):
        prevQ = self.getQvalue(prevstate, prevAction)
        diff = (reward + self.discount * self.getValue(state)
                - prevQ) * self.learningRate
        newQ = prevQ + diff
        # if all(prevstate == [7, 0,0]):
        # print("learn {} {} with {}".format(prevstate, prevAction, diff))
        self.q[prevstate][prevAction] = np.clip(newQ, -self.maxq, self.maxq)
        return diff

    def updateAndGetAction(self, reward, state, terminal):
        self.steps += 1
        if self.steps % 10000 == 0:
            self.dumpQ()

        if self.previousstate is None:
            return 0., self.getActionAndSaveState(state)

        if not self.previousstate == state or terminal:
            # print("newstate")
            self.history.append((self.previousstate, self.previousAction,
                                 state, reward))
            loss = self.updateQ(self.previousstate, self.previousAction, state, reward)
            return loss, self.getActionAndSaveState(state)
        else:
            # print("SAMEstate")
            # if same state, then repeate action
            return 0., self.possibleActions[self.previousAction]


    def updateHistory(self):
        """backward experience learning..."""
        # print("updating history of len={}".format(len(self.history)))
        for prevstate, prevA, state, reward in reversed(self.history):
            self.updateQ(prevstate, prevA, state, reward)
            self.history = []

class AutoDiscretizer(object):
    def __init__(self, size=10, minv=None, maxv=None):
        self.min = minv
        self.max = maxv
        self.size = size
        self.realmin = None
        self.realmax = None

    def normalise(self, value):
        # self.min = min(self.min, value) if self.min else value
        # self.max = max(self.max, value) if self.max else value
        # if self.max == self.min:
        # return 0
        if self.realmin is None:
            self.realmin = self.realmax = value
            self.realmin = min(self.realmin, value)
            self.realmax = max(self.realmax, value)
        if value >= self.max:
            return self.size - 1
        if value < self.min:
            return 0
        return int(self.size * (value - self.min) // (self.max - self.min))

def displayQ(q, ticks, folder):
    _, _, si, so = q.shape
    if not hasattr(displayQ, "fig"):
        displayQ.fig = plt.figure(1, (12., 6.))
        displayQ.grid = AxesGrid(displayQ.fig, 111,  # similar to subplot(111)
                                 nrows_ncols=(so, si),  # creates 2x2 grid of axes
                                 axes_pad=0.7,  # pad between axes in inch.
                                 share_all=True,
                                 cbar_location="right",
                                 cbar_mode="each",
                                 cbar_pad=0.1,
                                 label_mode="0",
                                 )
        displayQ.cbs = [None] * si  * (so)
        displayQ.ims = [None] * si  * (so)
    # vmin, vmax = (np.min(q), np.max(q))

    displayQ.fig.suptitle("Steps = {}".format(ticks))
    for i in range(si):
        for o in range(so):
            data = q[:, :, i, o]
            minmax = max(abs(data.min()), abs(data.max()))
            if displayQ.cbs[i+si*o] is None:
                # displayQ.grid[i+si*o].cla()
                displayQ.grid[i+si*o].set_title("Speed {} - {}".format(i, "Flap" if o else "No Flap"))
                displayQ.grid[i+si*o].set_xlabel("Distance to pipe")
                displayQ.grid[i+si*o].set_ylabel("Delta y to pipe")
                im = displayQ.grid[i+si*o].imshow(data.T, vmin=-minmax, vmax=minmax,
                                                  cmap=plt.get_cmap("seismic"),
                                                  origin="lower")
                displayQ.ims[i+si*o] = im
                displayQ.cbs[i+si*o] = displayQ.grid.cbar_axes[i+si*o].colorbar(im)
                displayQ.grid[i+si*o].set_xticks([])
                displayQ.grid[i+si*o].set_yticks([])
            else:
                displayQ.ims[i+si*o].set_data(data.T)
                displayQ.ims[i+si*o].set_clim(vmin=-minmax, vmax=minmax)

    displayQ.fig.savefig(
        os.path.join(folder, "Q-Values-{0:06d}.png".format(ticks)))
        # o = so
        # data = (q[:, :, i, 0] > q[:, :, i, 1]).astype(np.int)
        # minmax = max(abs(data.min()), abs(data.max()))
        # if displayQ.cbs[i+si*o] is None:
            # displayQ.grid[i+si*o].cla()
            # displayQ.grid[i+si*o].set_title("q[:, :, {}, {}]".format(i, o))
            # im = displayQ.grid[i+si*o].imshow(data.T, vmin=-minmax, vmax=minmax,
                                                # cmap=plt.get_cmap("seismic"),
                                                # origin="lower")
            # displayQ.ims[i+si*o] = im
            # displayQ.cbs[i+si*o] = displayQ.grid.cbar_axes[i*so+o].colorbar(im)
            # displayQ.grid[i+si*o].set_xticks([])
            # displayQ.grid[i+si*o].set_yticks([])
        # else:
            # displayQ.ims[i+si*o].set_data(data.T)
            # displayQ.ims[i+si*o].set_clim(vmin=-minmax, vmax=minmax)

    # plt.pause(0.05)



GRID_SIZE = 50
scalers = [AutoDiscretizer(size=GRID_SIZE, minv=-300, maxv=-1),
           AutoDiscretizer(size=GRID_SIZE, minv=-169, maxv=341),
           AutoDiscretizer(size=5, minv=-9, maxv=10),
           # AutoDiscretizer(size=GRID_SIZE, minv=-180, maxv=341),
           ]
def preprocessor(state):
    return np.array([
        scalers[0].normalise(-state["next_pipe_dist_to_player"]),
        scalers[1].normalise(state["next_pipe_bottom_y"] - state["player_y"]),
        scalers[2].normalise(state["player_vel"]),
        # scalers[3].normalise(state["next_next_pipe_bottom_y"] - state["player_y"]),
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

        reward = env.act(qAgent.getActionAndSaveState(tuple(env.getGameState())))

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

STEPS_PER_EPOCHS = 8000
EPOCHS = 400
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
qAgent = QAgent(env.getActionSet(), [s.size for s in scalers], discount=.99,
                       learningRate=.2, gridSize=GRID_SIZE, epsilon=epsilon)
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
    if e % 1 == 0:
        displayQ(qAgent.q, qAgent.steps, folder)
        for scale in scalers:
            print("scaler: min:{} max:{}".format(scale.realmin, scale.realmax))

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
        state = tuple(env.getGameState())
        loss, action = qAgent.updateAndGetAction(reward, state, env.game_over())

        avgloss += loss
        avgreward += reward

        # update epsilon
        epsilon += EPSILON_DECAY_V
        if epsilon < EPSILON_MIN:
            epsilon = EPSILON_MIN
        qAgent.updateEpsilon(epsilon)
        # clock.tick_busy_loop(FPS)

    if e % 1 == 0:
        nextTest = True

    print("Epoch {}: Steps:{} episodes:{} epsilon={:5.3f}, avgLoss={:5.3f},"
          "totalReward={:5.3f}, maxScore={}".format(
        e, qAgent.steps, qAgent.episodes,
        epsilon, avgloss / STEPS_PER_EPOCHS, avgreward, maxScore))
    print("avg Action {}".format(avgAction / STEPS_PER_EPOCHS))
plt.show()



