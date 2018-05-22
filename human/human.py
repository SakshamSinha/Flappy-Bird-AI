import numpy as np
from ple import PLE
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FlappyBirdClone import flappy
import pygame

###################################
game = flappy.FlappyClone()
env = PLE(game, display_screen=True, force_fps=False, fps=30)
env.init()

# press w to jump!
for f in range(15000):
    if env.game_over(): # if the game is over, reset
        print("tick {} death at score: {}".format(f, game.getScore()))
        env.reset_game()

    reward = env.act(None)
