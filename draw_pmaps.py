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

do = sys.argv[1]
outfile = sys.argv[2]
infiles = sys.argv[3:]

afs = map(asciifile.AsciiFile, infiles)

n_rows = len(afs)
n_cols = max(map(lambda af: len(af.params()), afs))

mpl.rcParams['image.cmap'] = 'gray'
mpl.rcParams['image.aspect'] = 'equal'
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
fig = plt.figure(figsize=(n_cols*6, n_rows*6))

for i, af in enumerate(afs):
    for j, a in enumerate(af.a.T):
        p = af.params()[j]
        ax = plt.subplot2grid((n_rows, n_cols), (i, j))
        ax.set_title(p)
        if j == 0:
            ax.set_ylabel(os.path.basename(af.filename))
        if do == 'pmap':
            img = a.reshape(af.subwinsize())
            plt.imshow(img)
            plt.colorbar()
        elif do == 'hist':
            plt.hist(a)

pylab.savefig(outfile, bbox_inches='tight')
