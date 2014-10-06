#!/usr/bin/env python2

# Draw comparative boxplots in one figure.

import os.path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab

from dwi import patient
from dwi import util

a, infiles = util.get_args(2)
outfile = a[0]

afs = map(asciifile.AsciiFile, infiles)
params = afs[0].params()

n_rows = 2
n_cols = len(params)

pmaps = np.array(map(lambda af: af.a.T, afs))
mins, maxs = pmaps.min(axis=0), pmaps.max(axis=0)

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
fig = plt.figure(figsize=(n_cols*6, n_rows*6))

for i in range(n_rows):
    for j in range(n_cols):
        ax = plt.subplot2grid((n_rows, n_cols), (i, j))
        ax.set_title(params[j])
        if j == 0:
            labels = map(lambda x: x.filename, afs[i::2])
            ax.set_ylabel('\n'.join(labels))
        x = map(lambda x: x[j], pmaps[i::2])
        plt.boxplot(x, vert=False, notch=False)
        r = (min(mins[j]), max(maxs[j]))
        ax.set_xlim(r[0], r[1])

pylab.savefig(outfile, bbox_inches='tight')
