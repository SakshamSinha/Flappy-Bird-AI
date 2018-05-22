""" Interface with the PLE environment

Authors: Vincent Francois-Lavet, David Taralla
Modified by: Norman Tasfi
"""
import numpy as np
import cv2
from ple import PLE
from ple.games.flappybird import FlappyBird

from deer.base_classes import Environment

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt


class MyEnv(Environment):
    VALIDATION_MODE = 0
    # original size is 288x512 so dividing

    def __init__(self, rng, game=None, frame_skip=4,
            ple_options={"display_screen": True, "force_fps":True, "fps":30}):

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._frame_skip = frame_skip if frame_skip >= 1 else 1
        self._random_state = rng
        self._hist_size = 1

        if game is None:
            raise ValueError("Game must be provided")


        self._ple = PLE(game, **ple_options)
        self._ple.init()

        self._actions = self._ple.getActionSet()
        self._state_size = self._ple.getGameStateDims()[0]
        self._state_saved = np.zeros((self._state_size), dtype=np.float32)
        self.previous_score = 0.
        self.episode_scores = []


    def reset(self, mode):
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self.episode_scores = []
                self.previous_score = .0
                self._mode_episode_count = 0
            else:
                self._mode_episode_count += 1
                self.episode_scores.append(self._mode_score - self.previous_score)
                self.previous_score = self._mode_score
        elif self._mode != -1: # and thus mode == -1
            self._mode = -1

        # print("Dead at score {}".format(self._ple.game.getScore()))
        self._ple.reset_game()
        for _ in range(self._random_state.randint(self._hist_size)):
             self._ple.act(self._ple.NOOP)

        return [[[0] * self._state_size] * self._hist_size]


    def act(self, action):
        action = self._actions[action]

        reward = 0
        for _ in range(self._frame_skip):
            reward += self._ple.act(action)

            if self.inTerminalState():
                break

        self._state_saved = self._ple.getGameState()
        self._mode_score += reward
        if self.inTerminalState():
            pass

        return reward #np.sign(reward)

    def summarizePerformance(self, test_data_set):
        if self.inTerminalState() == False:
            self._mode_episode_count += 1
        maxscore = max(self.episode_scores) if len(self.episode_scores) else "N/A"
        print("== Max score of episode is {} over {} episodes ==".format(
            maxscore, self._mode_episode_count))


    def inputDimensions(self):
        return [(self._hist_size, self._state_size)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return len(self._actions)

    def observe(self):
        return [np.array(self._state_saved)]

    def inTerminalState(self):
        return self._ple.game_over()



if __name__ == "__main__":
    pass
