# Author Corentin Chéron
# from random agent at
# https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/tests/test_ple.py

import numpy as np
from ple import PLE
import os, sys
import pygame
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FlappyBirdClone import flappy

# order of data in state
next_next_pipe_bottom_y = 0
next_next_pipe_dist_to_player = 1
next_next_pipe_top_y = 2
next_pipe_bottom_y = 3
next_pipe_dist_to_player = 4
next_pipe_top_y = 5
player_vel = 6
player_y = 7

noop = 1
flap = 0
PIPE_WIDTH = 52
BIRD_WIDTH = 20

V_MARGIN = 38
PIPE_DIST_DELTA = PIPE_WIDTH + BIRD_WIDTH + 5
DELTA_H = V_MARGIN - 15
PIPE_DIST_SKIP_UP = 12
PIPE_DIST_SKIP_DOWN = 12
DELTA = 3

FPS = 120
class ExpertAgent():
    """An approximate q-learning agent inspired from the Pacman project."""
    def __init__(self, actions, obsSize):
        # Takes a list of feature function as input
        self.possibleActions = actions
        self.possibleActionsInt = list(range(len(self.possibleActions)))
        # self.previousAction = noop
        self.pipeDelay = 0
        # self.previousPipeY = None


    def getAction(self, state):
        action = noop

        if state[next_pipe_bottom_y] > state[next_next_pipe_bottom_y]:
            # next pipe is higher
            if state[next_pipe_dist_to_player] < PIPE_DIST_SKIP_UP:
                # in advance jump
                selectedPipeBotY = state[next_next_pipe_bottom_y]
            else:
                selectedPipeBotY = state[next_pipe_bottom_y] - DELTA
        else:
            # next pipe is lower
            if state[next_pipe_dist_to_player] < PIPE_DIST_SKIP_DOWN:
                selectedPipeBotY = state[next_next_pipe_bottom_y]
            else:
                selectedPipeBotY = state[next_pipe_bottom_y] + DELTA

        if state[next_pipe_dist_to_player] < PIPE_DIST_DELTA:
            targetH = selectedPipeBotY - V_MARGIN
        else:
            targetH = selectedPipeBotY - DELTA_H

        if (state[player_y] > targetH):
            action = flap

        return self.possibleActions[action]


def preprocessor(state):
    return np.array([state[k] for k in sorted(state.keys())])


game = flappy.FlappyClone(crazy=False)
env = PLE(game, display_screen=True, force_fps=True, fps=30,
          state_preprocessor=preprocessor)
env.init()
expertAgent = ExpertAgent(env.getActionSet(), env.getGameStateDims())

for e in range(1, 101):
    while True:
        if env.game_over(): # if the game is over, reset
            print("test {}, death at score: {}".format(e, game.getScore()))
            env.game.tick(1. / 2.)
            env.reset_game()
            break
        reward = env.act(expertAgent.getAction(env.getGameState()))
        print("score={:010.1f}".format(game.getScore()), end="\r")
        env.game.tick(FPS)


