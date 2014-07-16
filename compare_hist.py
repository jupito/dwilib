#!/usr/bin/env python

# Draw parametric maps in one figure.

import os.path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab

from dwi import asciifile

if len(sys.argv) < 3:
    print 'Need parameters'
    sys.exit(1)

outfile = sys.argv[1]
infiles = sys.argv[2:]

afs = map(asciifile.AsciiFile, infiles)
params = afs[0].params()
shape = afs[0].subwinsize()

pmaps = np.array(map(lambda af: af.a.T, afs))
mins, maxs = pmaps.min(axis=0), pmaps.max(axis=0)

n_rows = len(pmaps) / 2
n_cols = len(params)

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
fig = plt.figure(figsize=(n_cols*6, n_rows*6))

for i in range(n_rows):
    for j in range(n_cols):
        ax = plt.subplot2grid((n_rows, n_cols), (i, j))
        ax.set_title(params[j])
        if j == 0:
            labels = (afs[2*i].filename, afs[2*i+1].filename)
            ax.set_ylabel('%s\n%s' % labels)
        x = [pmaps[2*i][j], pmaps[2*i+1][j]]
        r = (min(mins[j]), max(maxs[j]))
        plt.hist(x, range=r, rwidth=1)

pylab.savefig(outfile, bbox_inches='tight')
