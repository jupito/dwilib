#!/usr/bin/env python2

"""Compare ROI masks."""

import argparse

import numpy as np

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('-v', '--verbose',
            action='count',
            help='increase verbosity')
    p.add_argument('-s', '--subwindow', metavar='I',
            nargs=6, default=[], required=False, type=int,
            help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('mask1', metavar='FILE1',
            help='mask file #1')
    p.add_argument('mask2', metavar='FILE2',
            help='mask file #2')
    args = p.parse_args()
    return args

def read_mask_file(filename):
    arrays = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == '0' or line[0] == '1':
                a = np.array(list(line), dtype=int)
                arrays.append(a)
    mask = np.array(arrays)
    return mask

def mask_subwindow(mask, subwindow):
    subwindow = [n-1 for n in subwindow] # Handle 1-based indexing.
    z1, z2, y1, y2, x1, x2 = subwindow
    # XXX: Don't use Z axis (slice) for now.
    mask = mask[y1:y2, x1:x2]
    return mask

def roi_position(mask):
    # XXX: Quick and dirty ad hoc hack.
    for y, row in enumerate(mask):
        for x, n in enumerate(row):
            if n:
                return (y, x)

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

args = parse_args()
mask1 = read_mask_file(args.mask1)
mask2 = read_mask_file(args.mask2)

if args.subwindow:
    mask1 = mask_subwindow(mask1, map(int, args.subwindow))

if not mask1.shape == mask2.shape:
    raise Exception('Masks have different shapes.')

pos1 = roi_position(mask1)
pos2 = roi_position(mask2)

if args.verbose:
    print pos1, pos2
print distance(pos1, pos2)
