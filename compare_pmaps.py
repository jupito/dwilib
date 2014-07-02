#!/usr/bin/env python

# Draw parametric maps in one figure.

import os.path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab

import asciifile
import util

if len(sys.argv) < 3:
    print 'Need parameters'
    sys.exit(1)

do = sys.argv[1]
outfile = sys.argv[2]
infiles = sys.argv[3:]

afs = map(asciifile.AsciiFile, infiles)
params = afs[0].params()
shape = afs[0].subwinsize()

maps = np.array(map(lambda af: af.a.T, afs))
mins, maxs = maps.min(axis=0), maps.max(axis=0)
vars = maps.var(axis=0)
stddevs = np.sqrt(vars)

imagesets = maps
imagesetnames = map(lambda af: af.filename, afs)
#imagesets = [mins, maxs, stddevs]
#imagesetnames = ['min', 'max', 'stddev']

n_rows = len(imagesets)
n_cols = len(params)

mpl.rcParams['image.cmap'] = 'gray'
mpl.rcParams['image.aspect'] = 'equal'
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
fig = plt.figure(figsize=(n_cols*6, n_rows*6))

for i in range(n_rows):
    for j in range(n_cols):
        ax = plt.subplot2grid((n_rows, n_cols), (i, j))
        ax.set_title(params[j])
        if j == 0:
            ax.set_ylabel(imagesetnames[i])
        a = imagesets[i][j]
        r = (min(mins[j]), max(maxs[j]))
        if do == 'pmap':
            img = a.reshape(shape)
            plt.imshow(img, vmin=r[0], vmax=r[1])
            plt.colorbar()
        elif do == 'hist':
            plt.hist(a, range=r)

pylab.savefig(outfile, bbox_inches='tight')
