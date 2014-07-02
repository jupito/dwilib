#!/usr/bin/env python

# Plot by gleason score.

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

outfile = sys.argv[1]
infiles = sys.argv[2:]

scores = asciifile.read_gleason_file(infiles[0])
afs = map(asciifile.AsciiFile, infiles[1:])
params = afs[0].params()

#scores.sort(key=lambda x: (-x[2], -x[3], x[0]))
#for score in scores:
#    print score
#for af in afs:
#    print '%s\t%s' % (af.filename, asciifile.get_gs(scores, af.basename))

n_rows = 1
n_cols = len(params)

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['image.cmap'] = 'gray'
mpl.rcParams['image.aspect'] = 'equal'
mpl.rcParams['image.interpolation'] = 'none'
fig = plt.figure(figsize=(n_cols*6, n_rows*6))

for col in range(n_cols):
    ax = plt.subplot2grid((n_rows, n_cols), (0, col))
    ax.set_title(params[col])
    x = [[], []]
    y = [[], []]
    for af in afs:
        score = asciifile.get_gs(scores, af.basename)
        if score:
            ps = af.a.T[col]
            n = af.number()-1
            for p in ps:
                x[n].append(2*score[2] + score[3])
                y[n].append(p)
        else:
            print '%s has no score' % af.basename
    #plt.scatter(x[0], y[0], c='b', marker='x')
    #plt.scatter(x[1], y[1], c='g', marker='x')
    plt.hexbin(x[0], y[0], gridsize=10)

pylab.savefig(outfile, bbox_inches='tight')
