#!/usr/bin/env python2

"""test"""

import argparse
import numpy as np

import dwi.dwimage
import dwi.util

EPSILON = 1e-6

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='increase verbosity')
    p.add_argument('--input1', '-i1', metavar='FILENAME', required=True,
            nargs='+', default=[], help='input ASCII files 1')
    p.add_argument('--input2', '-i2', metavar='FILENAME', required=True,
            nargs='+', default=[], help='input ASCII files 2')
    args = p.parse_args()
    return args

def read_img(filename):
    img = dwi.dwimage.load(filename)[0].sis
    img = np.sum(img, axis=0)
    return img

def distance_hist(hist1, hist2):
    """Histogram intersection distance measure."""
    pairs = np.array([hist1, hist2]).T
    return sum(min(pair) for pair in pairs)

def distance_log(hist1, hist2, eps=EPSILON):
    """Log-likelihood distance measure."""
    pairs = np.array([hist1, hist2]).T
    return -sum(a*np.log(max(b, eps)) for a, b in pairs)

def distance_chi(hist1, hist2, eps=EPSILON):
    """Chi-squared distance measure."""
    pairs = np.array([hist1, hist2]).T
    return sum((a-b)**2/(max(a+b, eps)) for a, b in pairs)


args = parse_args()
imgs1 = [read_img(f) for f in args.input1]
imgs2 = [read_img(f) for f in args.input2]
imgs = np.array([imgs1, imgs2])
imgs = np.sum(imgs, axis=1)
print imgs.shape

distances = [distance_hist(*t) for t in zip(imgs1, imgs2)]
print dwi.util.fivenum(distances)

distances = [distance_log(*t) for t in zip(imgs1, imgs2)]
print dwi.util.fivenum(distances)

distances = [distance_chi(*t) for t in zip(imgs1, imgs2)]
print dwi.util.fivenum(distances)

#import matplotlib.pyplot as pl
#pl.bar(np.arange(0, 10), imgs[0], width=0.4, color='r')
#pl.bar(np.arange(0, 10)+0.4, imgs[1], width=0.4, color='g')
#pl.show()
#pl.plot(np.arange(0, 10), imgs[0], color='r')
#pl.plot(np.arange(0, 10), imgs[1], color='g')
#pl.show()
