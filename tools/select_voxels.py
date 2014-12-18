#!/usr/bin/env python2

"""Select voxels from image by subwindow, ROI, and write them into an ASCII
file."""

import argparse
import numpy as np

import dwi.dwimage
import dwi.mask
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description = __doc__)
    p.add_argument('-v', '--verbose',
            action='count',
            help='increase verbosity')
    p.add_argument('--subwindow', '-s', metavar='I',
            nargs=6, default=[], required=False, type=int,
            help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('--mask', '-m', metavar='MASKFILE',
            required=False,
            help='mask file (applied within subwindow size)')
    p.add_argument('--input', '-i', metavar='INFILE',
            required=True,
            help='input parametric map file')
    p.add_argument('--output', '-o', metavar='OUTFILE',
            required=True,
            help='output parametric map file')
    args = p.parse_args()
    return args

def write_pmap_ascii_head(dwi, model, params, f):
    f.write('subwindow: [%s]\n' % ' '.join(map(str, dwi.subwindow)))
    f.write('number: %d\n' % dwi.number)
    f.write('bset: [%s]\n' % ' '.join(map(str, dwi.bset)))
    f.write('ROIslice: %s\n' % dwi.roislice)
    f.write('name: %s\n' % dwi.name)
    f.write('executiontime: %d s\n' % dwi.execution_time())
    f.write('description: %s %s\n' % (dwi.filename, repr(model)))
    f.write('model: %s\n' % model)
    f.write('parameters: %s\n' % ' '.join(map(str, params)))

def write_pmap_ascii_body(pmap, f):
    for p in pmap:
        f.write(' '.join(map(repr, p)) + '\n')

args = parse_args()

# Load image.
dwimage = dwi.dwimage.load(args.input)[0]
if args.verbose:
    print dwimage

# Select subwindow.
if args.subwindow:
    if args.verbose:
        print 'Using subwindow %s' % args.subwindow
    dwimage = dwimage.get_roi(args.subwindow, onebased=True)

# Select sequence of voxels.
image = dwimage.image
if args.mask:
    mask = dwi.mask.load_ascii(args.mask)
    if args.verbose:
        print 'Using mask %s' % mask
    voxels = mask.get_masked(image)
else:
    voxels = image.reshape((-1,image.shape[-1]))

# Output voxels.
if args.verbose:
    print 'Writing %i voxels with %i values to %s' % (voxels.shape[0],
            voxels.shape[1], args.output)

with open(args.output, 'w') as f:
    model = 'selection'
    params = range(voxels.shape[1])
    write_pmap_ascii_head(dwimage, model, params, f)
    write_pmap_ascii_body(voxels, f)
