"""ALE launcher. See Wiki for more details about this experiment.

Authors: Vincent Francois-Lavet, David Taralla

Modified by Corentin Chéron
"""

import sys
import logging
import numpy as np
from joblib import hash, dump, load
import os

from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.q_networks.q_net_keras import MyQNetwork
# from qNet import NN as myNN
from PLE_env import MyEnv as PLE_env
from qNet import NN as myNN
import deer.experiment.base_controllers as bc
from matplotlib import pyplot as plt
from deer.policies import EpsilonGreedyPolicy, LongerExplorationPolicy

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from FlappyBirdClone import flappy


class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 2000
    EPOCHS = 50
    STEPS_PER_TEST = 2000
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    # frame_skip: int (default: 1)
    # The number of times we skip getting observations while
    # repeat an action.
    FRAME_SKIP = 4

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    BATCH_SIZE = 32
    CLIP_DELTA = 1.0
    DETERMINISTIC = True
    DISCOUNT = 0.999
    DISCOUNT_INC = 1
    DISCOUNT_MAX = 0.999
    EPSILON_START = .1
    EPSILON_MIN = .01
    EPSILON_DECAY = STEPS_PER_EPOCH * EPOCHS
    FREEZE_INTERVAL = 1000
    LEARNING_RATE = 0.025
    LEARNING_RATE_DECAY = 0.99
    MOMENTUM = 0.95
    REPLAY_MEMORY_SIZE = 20000
    RMS_DECAY = 0.3
    RMS_EPSILON = 0.0001
    UPDATE_FREQUENCY = 1
    UPDATE_RULE = 'sgd'
    # NB_TRAIN_PER_EPOCH = REPLAY_MEMORY_SIZE // BATCH_SIZE // 10
    NB_TRAIN_PER_EPOCH = 1


class AutoDiscretizer(object):
    def __init__(self, minv=None, maxv=None):
        self.min = minv
        self.max = maxv
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
        return (value - self.min) / (self.max - self.min)


scalers = [AutoDiscretizer(minv=-300, maxv=-1),
           AutoDiscretizer(minv=-169, maxv=341),
           AutoDiscretizer(minv=-9, maxv=10),
           AutoDiscretizer(minv=-200, maxv=200),
           ]
def preprocessor(state):
    return np.array([
        scalers[0].normalise(-state["next_pipe_dist_to_player"]),
        scalers[1].normalise(state["next_pipe_bottom_y"] - state["player_y"]),
        scalers[2].normalise(state["player_vel"]),
        scalers[3].normalise(state["next_next_pipe_bottom_y"] - state["next_pipe_bottom_y"]),
    ])

def preprocessor(state):
    return np.array([
        state[k] for k in sorted(state.keys())
    ])

if __name__ == "__main__":
    test = False
    game = flappy.FlappyClone(black=True)
    # game = Catcher(width=64, height=64)
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(523456)
    else:
        rng = np.random.RandomState()

    # --- Instantiate environment ---
    env = PLE_env(rng, game=game, frame_skip=parameters.frame_skip,
            ple_options={"display_screen": True, "force_fps":not test, "fps":30,
                         "rng":rng, "state_preprocessor":preprocessor})

    game.rewards["tick"] = 0.00001
    # --- Instantiate qnetwork ---
    qnetwork = MyQNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_delta,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng,
        neural_network=myNN)

    test_policy = EpsilonGreedyPolicy(qnetwork, env.nActions(), rng, 0.00)

    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng,
        test_policy=test_policy)

    # --- Create unique filename for FindBestController ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "PLE_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(
        evaluate_on='epoch',
        periodicity=1))

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(bc.TrainerController(
        evaluate_on='action',
        periodicity=parameters.update_frequency,
        show_episode_avg_V_value=False,
        show_avg_Bellman_residual=False,
        nb_train=parameters.nb_train_per_epoch))

    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate,
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))

    # Same for the discount factor.
    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        periodicity=1))

    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
    # episode or epoch (or never, hence the resetEvery='none').
    agent.attach(bc.EpsilonController(
        initial_e=parameters.epsilon_start,
        e_decays=parameters.epsilon_decay,
        e_min=parameters.epsilon_min,
        evaluate_on='action',
        periodicity=1,
        reset_every='none'))

    # We wish to discover, among all versions of our neural network (i.e., after every training epoch), which one
    # seems to generalize the better, thus which one has the highest validation score. Here, we do not care about the
    # "true generalization score", or "test score".
    # To achieve this goal, one can use the FindBestController along with an InterleavedTestEpochControllers. It is
    # important that the validationID is the same than the id argument of the InterleavedTestEpochController.
    # The FindBestController will dump on disk the validation scores for each and every network, as well as the
    # structure of the neural network having the best validation score. These dumps can then used to plot the evolution
    # of the validation and test scores (see below) or simply recover the resulting neural network for your
    # application.
    agent.attach(bc.FindBestController(
        validationID=PLE_env.VALIDATION_MODE,
        testID=None,
        unique_fname=fname))

    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a
    # "validation epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want
    # these validation epoch to interfere with the training of the agent, which is well established by the
    # TrainerController, EpsilonController and alike. Therefore, we will disable these controllers for the whole
    # duration of the validation epochs interleaved this way, using the controllersToDisable argument of the
    # InterleavedTestEpochController. For each validation epoch, we want also to display the sum of all rewards
    # obtained, hence the showScore=True. Finally, we want to call the summarizePerformance method of ALE_env every
    # [parameters.period_btw_summary_perfs] *validation* epochs.
    agent.attach(bc.InterleavedTestEpochController(
        id=PLE_env.VALIDATION_MODE,
        epoch_length=parameters.steps_per_test,
        controllers_to_disable=[0, 1, 2, 3, 4],
        periodicity=2,
        show_score=True,
        summarize_every=1))

    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass

    # handle loading / saving weights
    savedPath = fname + "_final"
    if os.path.exists("nnets/" + savedPath): # ugly, but as in dumpNetwork
        print("Loading saved net: " + savedPath)
        agent.setNetwork(savedPath)
    if test:
        agent.startMode(PLE_env.VALIDATION_MODE, 10000)

    dump(vars(parameters), "params/" + fname + ".jldump")
    agent.run(parameters.epochs, parameters.steps_per_epoch)
    if test:
        agent.summarizeTestPerformance()

    else:
        # -- save network
        agent.dumpNetwork(savedPath)

    # --- Show results ---
    basename = "scores/" + fname
    scores = load(basename + "_scores.jldump")
    plt.plot(range(1, len(scores['vs'])+1), scores['vs'], label="VS", color='b')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Score")
    plt.savefig(basename + "_scores.pdf")
    plt.show()
