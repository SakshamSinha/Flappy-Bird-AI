#!/usr/bin/python
# coding=utf-8

# Base Python File (plot-q.py)
# Created: lun. 18 déc. 2017 15:45:19 GMT
# Version: 1.0
#
# This Python script was developped by Cory.
#
# (c) Cory <sgryco@gmail.com>

import sys
import logging
import numpy as np
from joblib import hash, dump, load
import os
import datetime, time
from matplotlib import pyplot as plt


# --- Show results ---
basename = "scores/PLE_" + sys.argv[1]
scores = load(basename + "_scores.jldump")
plt.scatter(range(1, len(scores['vs'])+1), scores['vs'], label="Average Score", marker='x')
plt.legend()
plt.xlabel("Number of epochs")
plt.ylabel("Score")
mainTitle = "Deep Q-learning"
st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
plt.savefig(st + sys.argv[1].replace(" ", "_") + ".pdf")
