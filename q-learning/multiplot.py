#!/usr/bin/python
# coding=utf-8

# Base Python File (plot.py)
# Created: lun. 11 déc. 2017 09:55:21 GMT
# Version: 1.0
#
# This Python script was developped by Cory.
#
# (c) Cory <sgryco@gmail.com>


import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from joblib import dump, load
import datetime, time
import itertools

def loaddata(data):
    #loaded q, steps, stats
    eps = []
    scs = []
    for d in data:
        q, steps, episodes, stats = load(d + "/q-agent2-qvalues.jldump")

        steps, episodes, scores = zip(*stats)
        # BE CAREFUL, WE ADD 5 here to make it the max score reached and
        # prevent negative values
        scores = np.array(scores, dtype=np.int) + 5
        avgScores = scores[:, 1]
        maxScores = scores[:, 0]
        epochs = np.array(steps, dtype=np.int) // 2000 # convert steps in epochs normalized by 2k steps
        # N = 10
        # avgScores = np.convolve(avgScores, np.ones((N,))/N, mode='valid')
        eps.append(epochs)
        scs.append(avgScores)
        # scs.append(maxScores)
    return eps, scs

if len(sys.argv) < 3:
    print("error argument <jlib> <graph-description> required")
    sys.exit(-1)

data = sys.argv[1:]

epochs, scores = loaddata(data)
dsc = [
     "base",
    "Test A",
    "Test B",
    "Test C",
    "Test D",
    "Test E",
     # "learning_rate=0.02",
     # "learning_rate=0.005",
     # "epsilon_start=1.0",
     # "epsilon_start=0.5",
     # "epsilon_start=0.2",
     # "grid_size=50",
     # "grid_size=25",
     # "grid_size=10",
    # "Vspeed_size=10",
    # "Vspeed_size=5",
    # "Vspeed_size=3",
    # "Vspeed_size=2",
]
mainTitle = "Comparison of parameter sets"
# mainTitle = "Comparison of Dx and Dy discretisation grid size"
# mainTitle = "Comparison of starting epsilon values"
# mainTitle = "Comparison of number of discrete speeds"
marker = itertools.cycle(('x', '+', '.', 'o', '*'))
fig = plt.figure()
# plt.plot(epochs, avgScore, label="Discrete Q-Agent (lr=0.2)", color='b')
fig.suptitle(mainTitle)
for i in range(len(epochs)):

    plt.scatter(epochs[i], scores[i], label=dsc[i], marker=next(marker))
             # linewidth=.1)

plt.legend()
plt.xlabel("Training Epochs")
plt.ylabel("Average score on 100 games")
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
plt.savefig(st + mainTitle.replace(" ", "_") + ".pdf")
# plt.show()


