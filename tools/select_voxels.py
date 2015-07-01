#!/usr/bin/env python2

"""Select voxels from image by subwindow, ROI, and write them into an ASCII
file. Multiple same-size images may be combined by overlaying the parameters."""

from __future__ import division, print_function
import argparse
import collections
import os
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
    p.add_argument('--input', '-i', metavar='INFILE',
            nargs='+', required=True,
            help='input parametric map files')
    p.add_argument('--subwindow', '-s', metavar='I',
            nargs=6, default=[], required=False, type=int,
            help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('--mask', '-m', metavar='MASKFILE',
            required=False,
            help='mask file (applied within subwindow size)')
    p.add_argument('--keep-masked', action='store_true',
            help='keep masked voxels (as zeros)')
    p.add_argument('--subwindow-mask', action='store_true',
            help='apply subwindow on mask, too')
    p.add_argument('--output', '-o', metavar='OUTFILE',
            help='output parametric map file')
    args = p.parse_args()
    return args

def merge_dwimages(dwimages):
    # Merge multiple images of same size by overlaying the parameters.
    images = [d.image for d in dwimages]
    bsets = [d.bset for d in dwimages]
    filenames = [d.filename for d in dwimages]
    image = np.concatenate(images, axis=-1)
    bset = np.concatenate(bsets)
    dwimage = dwi.dwimage.DWImage(image, bset)
    dwimage.filename = '; '.join(filenames)
    dwimage.number = dwimages[0].number
    dwimage.subwindow = dwimages[0].subwindow
    dwimage.roislice = dwimages[0].roislice
    dwimage.name = dwimages[0].name
    dwimage.voxel_spacing = dwimages[0].voxel_spacing
    return dwimage

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

def ext(filename):
    """Return filename extension without the leading dot."""
    _, ext = os.path.splitext(filename)
    return ext[1:]

def write(filename, dwimage, image, fmt=None):
    """Write output file."""
    params = range(image.shape[-1])
    if fmt is None:
        fmt = ext(filename)
    if fmt in ['hdf5', 'h5']:
        import dwi.hdf5
        attrs = collections.OrderedDict()
        attrs['bset'] = dwimage.bset
        attrs['parameters'] = params
        dwi.hdf5.write_hdf5(filename, image, attrs)
    elif fmt in ['txt', 'ascii']:
        image = image.reshape((-1, image.shape[-1]))
        with open(filename, 'w') as f:
            write_pmap_ascii_head(dwimage, 'selection', params, f)
            write_pmap_ascii_body(image, f)
    else:
        raise Exception('Unknown format: {}'.format(fmt))

args = parse_args()

# Load image.
if len(args.input) == 1:
    dwimage = dwi.dwimage.load(args.input[0])[0]
else:
    dwimages = []
    for infile in args.input:
        dwimages.append(dwi.dwimage.load(infile)[0])
    dwimage = merge_dwimages(dwimages)
if args.verbose:
    print(dwimage)

# Select subwindow.
if args.subwindow:
    if args.verbose:
        print('Using subwindow %s' % args.subwindow)
    dwimage = dwimage.get_roi(args.subwindow, onebased=True)

# Select sequence of voxels.
image = dwimage.image
if args.mask:
    mask = dwi.mask.read_mask(args.mask)
    if args.subwindow and args.subwindow_mask:
        mask = mask.get_subwindow(args.subwindow)
    if args.verbose:
        print('Using mask %s' % mask)
    if args.keep_masked:
        image = mask.apply_mask(image)
    else:
        image = mask.selected(image)

# Write output voxels. Unless output filename is specified, one will be
# constructed from (first) input filename.
outfile = args.output or os.path.basename(args.input[0]) + '.txt'
if args.verbose:
    print('Writing %i voxels with %i parameters to %s' % (image.size,
            image.shape[-1], outfile))
write(outfile, dwimage, image)
