#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

# Write plots about DW images.

import sys
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from dwi import dwimage

PREFIX = 'plot'

def draw_plot(dwi, vx, vy):
    from pylab import plot, ylim, yticks
    
    t = np.arange(dwi.number)

    # data
    s = dwi.sis[:, vx, vy]
    plot(t, s, 'k+')

    # ADCm
    popt, err = dwi.fit_elem('adcm', vx, vy)
    print popt, err
    f = lambda b: dwimage.Functions['adcm'](b, *popt)
    s = np.array([f(b) for b in dwi.bset])
    plot(t, s, 'r-')

    # biexp
    popt, err = dwi.fit_elem('biexp', vx, vy)
    print popt, err
    f = lambda b: dwimage.Functions['biexp'](b, *popt)
    s = np.array([f(b) for b in dwi.bset])
    plot(t, s, 'g-')
    
    #ylim(0, max(max(s1), max(s2), max(s3)))
    #yticks(np.arange(4), ['s1', 's2', 's3', 's4'])


dwi = dwimage.DWImage('si_mice_data_lowb/diff_smallb_m1w1.mat', 'ROIdata')
#dwi = dwimage.DWImage('si_mice_data_highb/diff_largeb_m1w1.mat', 'ROIdata')
print dwi

#fig = plt.figure()
#fig.add_subplot(131)
#draw_plot(dwi, 15, 15)
#fig.add_subplot(132)
#draw_plot(dwi, 20, 20)
#fig.add_subplot(133)
#draw_plot(dwi, 25, 25)
#plt.tight_layout()
#plt.show()

for x, y in [(15, 15), (20, 20), (25, 25)]:
    fig = plt.figure()
    draw_plot(dwi, x, y)
    filename = '%s_%i,%i.png' % (PREFIX, x, y)
    print 'Writing %s...' % filename
    mpl.pylab.savefig(filename, bbox_inches='tight')
