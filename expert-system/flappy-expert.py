import numpy as np
from ple import PLE
from ple.games import flappybird

class NaiveAgent():
    """
        This is our naive agent. It picks actions at random!
    """
    def __init__(self, actions):
        self.actions = actions

    def pickAction(self, reward, obs=None):
        player_y = obs[0]["player_y"]
        print("player y={}".format(player_y), obs[0])
        if player_y > 150:
            return self.actions[0]
        else:
            return None
        # return self.actions[np.random.randint(0, len(self.actions))]

###################################
game = flappybird.FlappyBird()
env = PLE(game, state_preprocessor=lambda x: np.array(x).flatten())
agent = NaiveAgent(env.getActionSet())
env.init()
env.display_screen = True
env.force_fps = False

reward = 0.0
for f in range(15000):
    #if the game is over
        if env.game_over():
            env.reset_game()

        action = agent.pickAction(reward, env.getGameState())
        reward = env.act(action)

        # if f > 2000:
            # env.force_fps = False

        # if f > 2250:
            # env.display_screen = True
            # env.force_fps = True
