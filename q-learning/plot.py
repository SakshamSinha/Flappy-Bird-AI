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

if len(sys.argv) < 3:
    print("error argument <jlib> <graph-description> required")
    sys.exit(-1)

#loaded q, steps, stats
q, steps, stats = load(sys.argv[1])
print("loaded Q-Values from previous steps={}".format(steps))

epochs, avgScore = zip(*stats)
# BE CAREFUL, WE ADD 5 here to make it the max score reached and
# prevent negative values
avgScore = np.array(avgScore, dtype=np.int) + 5
epochs = np.array(epochs, dtype=np.int) // 2000 # convert steps in epochs normalized by 2k steps

fig = plt.figure()
# plt.plot(epochs, avgScore, label="Discrete Q-Agent (lr=0.2)", color='b')
plt.plot(epochs, avgScore, label=sys.argv[2], color='b')
plt.legend()
plt.xlabel("Epochs (2000 steps)")
plt.ylabel("Average score on 50 games")
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
plt.savefig(st + sys.argv[2] + ".pdf")
plt.show()


