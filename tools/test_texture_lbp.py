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

def lbpf_dist(hist1, hist2, method='chi-squared', eps=EPSILON):
    """Measure the distance of two LBP frequency histograms.
    
    Method can be one of the following:
    intersection: histogram intersection distance measure
    log-likelihood: log-likelihood distance measure
    chi-squared: distance measure
    """
    pairs = np.array([hist1, hist2]).T
    if method == 'intersection':
        r = sum(min(pair) for pair in pairs)
    elif method == 'log-likelihood':
        r = -sum(a*np.log(max(b, eps)) for a, b in pairs)
    elif method == 'chi-squared':
        r = sum((a-b)**2/(max(a+b, eps)) for a, b in pairs)
    else:
        raise Exception('Unknown distance measure: %s' % method)
    return r


args = parse_args()
imgs1 = [read_img(f) for f in args.input1]
imgs2 = [read_img(f) for f in args.input2]
imgs = np.array([imgs1, imgs2])
imgs = np.sum(imgs, axis=1)
print imgs.shape

for method in 'intersection log-likelihood chi-squared'.split():
    print method
    distances = [lbpf_dist(a, b, method) for a, b in zip(imgs1, imgs2)]
    print dwi.util.fivenum(distances)

#import matplotlib.pyplot as pl
#pl.bar(np.arange(0, 10), imgs[0], width=0.4, color='r')
#pl.bar(np.arange(0, 10)+0.4, imgs[1], width=0.4, color='g')
#pl.show()
#pl.plot(np.arange(0, 10), imgs[0], color='r')
#pl.plot(np.arange(0, 10), imgs[1], color='g')
#pl.show()
