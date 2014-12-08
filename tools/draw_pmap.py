#!/usr/bin/env python2

# -*- coding: iso-8859-15 -*-

# Draw parametric maps and their histograms.

import os.path
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pylab

from dwi import asciifile

def draw_pmap(img, filename):
    fig = plt.figure()
    #mpl.rcParams['image.cmap'] = 'hot'
    mpl.rcParams['image.cmap'] = 'gray'
    mpl.rcParams['image.aspect'] = 'equal'
    mpl.rcParams['image.interpolation'] = 'none'
    plt.imshow(img)
    print 'Writing figure %s...' % filename
    pylab.savefig(filename, bbox_inches='tight')

def draw_histogram(array, filename):
    fig = plt.figure()
    plt.hist(array)
    print 'Writing figure %s...' % filename
    pylab.savefig(filename, bbox_inches='tight')

for filename in sys.argv[1:]:
    af = asciifile.AsciiFile(filename)
    for i, a in enumerate(af.a.T):
        p = af.params()[i]
        name = '%s_%s_%02i%s.png' % (os.path.basename(filename), 'pmap', i, p)
        draw_pmap(a.reshape(af.subwinsize()[1:]), name)
        name = '%s_%s_%02i%s.png' % (os.path.basename(filename), 'hist', i, p)
        draw_histogram(a, name)
