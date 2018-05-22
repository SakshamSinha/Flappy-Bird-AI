import numpy as np
from ple import PLE
from ple.games import flappybird
import pygame

class HumanAgent():
    """
        This is our naive agent. It picks actions at random!
    """
    def __init__(self, actions):
        self.actions = actions + [None]

    def pickAction(self, reward, obs=None):
        flapped = False
        for event in pygame.event.get():
            print("event:", event)
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_UP):
                print("key up")
                flapped = True
        return self.actions[not flapped]

###################################
game = flappybird.FlappyBird()
# env = PLE(game, state_preprocessor=lambda x: np.array(x).flatten())
pygame.init()
clock = pygame.time.Clock()
game.rng = np.random.RandomState(123456)
game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
game.init()
# pygame.event.set_grab(True) ## locks everything
# env.display_screen = True
# env.force_fps = False
fps = 30
reward = 0.0

# agent = HumanAgent(game.actions)
for f in range(15000):
    #if the game is over
    if game.game_over():
        game.init()

    # action = agent.pickAction(reward, env.getGameState())
    # reward = env.act(action)
    game.step(1000. / fps)

    clock.tick(fps)
    pygame.event.pump()
    pygame.display.flip()
        # if f > 2000:
            # env.force_fps = False

        # if f > 2250:
            # env.display_screen = True
            # env.force_fps = True
