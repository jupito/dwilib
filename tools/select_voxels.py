#!/usr/bin/env python2

"""Select voxels from image and write them into another file. Output may
include all voxels, or a selection made by subwindow specification or mask
file.

Multiple same-size images may be combined by overlaying the parameters. In this
case, the output file will receive its attributes from the first input file,
except for the 'parameters' attribute, which will be aggregated.
"""

from __future__ import absolute_import, division, print_function
import argparse
from operator import add
import os.path

import numpy as np

import dwi.files
import dwi.mask
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-v', '--verbose', action='count',
                   help='increase verbosity')
    p.add_argument('--input', '-i', metavar='INFILE', nargs='+', required=True,
                   help='input parametric map files')
    p.add_argument('--subwindow', '-s', metavar='I', nargs=6, default=[],
                   required=False, type=int,
                   help='use subwindow (specified by 6 one-based indices)')
    p.add_argument('--mask', '-m', metavar='MASKFILE', required=False,
                   help='mask file (applied within subwindow size)')
    p.add_argument('--keep-masked', action='store_true',
                   help='keep masked voxels (as zeros)')
    p.add_argument('--subwindow-mask', action='store_true',
                   help='apply subwindow on mask, too')
    p.add_argument('--output', '-o', metavar='OUTFILE',
                   help='output parametric map file')
    return p.parse_args()


def main():
    args = parse_args()

    # Read and merge images.
    tuples = [dwi.files.read_pmap(x) for x in args.input]
    image = np.concatenate([x for x, _ in tuples], axis=-1)
    attrs = tuples[0][1]
    attrs['parameters'] = reduce(add, (x['parameters'] for _, x in tuples))
    if args.verbose:
        print(image.shape)
        print(attrs)

    # Select subwindow.
    if args.subwindow:
        if args.verbose:
            print('Using subwindow %s' % args.subwindow)
        image = dwi.util.crop_image(image, args.subwindow,
                                    onebased=True).copy()

    # Apply mask.
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
        print('Writing {nv} voxels with {np} parameters to {of}'.format(
            nv=image.size, np=image.shape[-1], of=outfile))
    dwi.files.write_pmap(outfile, image, attrs)


if __name__ == '__main__':
    main()
