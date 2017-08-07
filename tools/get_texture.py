#!/usr/bin/python3

"""Calculate texture properties for a masked area."""

import argparse
from collections import defaultdict
import logging

import numpy as np

import dwi.files
import dwi.mask
import dwi.standardize
import dwi.texture
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--input', required=True,
                   help='input image')
    p.add_argument('--mask',
                   help='mask file to use')
    p.add_argument('--mode', metavar='MODE', required=True,
                   help='imaging mode specification')
    p.add_argument('--method', metavar='METHOD', required=True,
                   help='method')
    p.add_argument('--slices', default='maxfirst',
                   help='slice selection (maxfirst, max, all)')
    p.add_argument('--winspec', default='5',
                   help='window specification (side length, all, mbb)')
    p.add_argument('--portion', type=float, default=0,
                   help='portion of selected voxels required for each window')
    p.add_argument('--voxel', choices=('all', 'mean', 'median'), default='all',
                   help='voxel to output (all, mean, median)')
    p.add_argument('--output', metavar='FILENAME', required=True,
                   help='output texture map file')
    return p.parse_args()


def max_mask(mask, winsize):
    """Return a mask that has the voxels selected that have the maximum number
    of surrounding voxels selected in the original mask.
    """
    d = defaultdict(list)
    for pos, win in dwi.util.sliding_window(mask, winsize, mask=mask):
        d[np.count_nonzero(win)].append(pos)
    r = np.zeros_like(mask)
    for pos in d[max(d)]:
        r[pos] = True
    return r


def portion_mask(mask, winsize, portion=1, resort_to_max=True):
    """Return a mask that selects (only) voxels that have the window at each
    selected voxel origin up to a minimum portion in the original mask selected
    (1 means the whole window must be selected, 0 gives the original mask).

    If resort_to_max is true, the window with maximum number of selected voxels
    is used in case the resulting mask would otherwise be empty.
    """
    r = np.zeros_like(mask)
    for pos, win in dwi.util.sliding_window(mask, winsize, mask=mask):
        if np.count_nonzero(win) / win.size >= portion:
            r[pos] = True
    if resort_to_max and np.count_nonzero(r) == 0:
        r = max_mask(mask, winsize)
    return r


def main():
    args = parse_args()
    loglevel = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=loglevel, stream=logging.sys.stdout)

    logging.info('Reading image: %s', args.input)
    img, attrs = dwi.files.read_pmap(args.input)
    if args.mode == 'T2':
        assert attrs['echotimes'][0] == 0  # TODO: Could be another?
    img = img[..., 0]
    assert img.ndim == 3

    if args.mask is None:
        mask = np.zeros_like(img, dtype=np.bool)
        mbb = dwi.util.bbox(img, pad=0)
        logging.info('MBB mask: %s', mbb)
        mask[mbb] = True
        mask = dwi.mask.Mask3D(mask)
    else:
        logging.info('Using mask: %s', args.mask)
        mask = dwi.mask.read_mask(args.mask)
        if isinstance(mask, dwi.mask.Mask):
            mask = mask.convert_to_3d(img.shape[0])

    if img.shape != mask.shape():
        raise Exception('Image shape {} does not match mask shape {}'.format(
            img.shape, mask.shape()))
    if mask.n_selected() == 0:
        raise ValueError('Empty mask.')

    if args.slices == 'maxfirst':
        slice_indices = [mask.max_slices()[0]]
    elif args.slices == 'max':
        slice_indices = mask.max_slices()
    elif args.slices == 'all':
        slice_indices = mask.selected_slices()
    else:
        raise Exception('Invalid slice set specification', args.slices)

    # Zero other slices in mask.
    for i in range(len(mask.array)):
        if i not in slice_indices:
            mask.array[i, :, :] = 0

    # Use only selected slices to save memory.
    if not args.voxel == 'all':
        img = img[slice_indices].copy()
        mask.array = mask.array[slice_indices].copy()

    # Get portion mask.
    if args.winspec in ('all', 'mbb'):
        pmask = mask.array  # Some methods don't use window.
    elif args.winspec.isdigit():
        winsize = int(args.winspec)
        assert winsize > 0
        winshape = (1, winsize, winsize)
        pmask = portion_mask(mask.array, winshape, portion=args.portion)
    else:
        raise ValueError('Invalid window spec: {}'.format(args.winspec))

    logging.info('Image: %s, slice: %s, voxels: %s, window: %s', img.shape,
                 slice_indices, np.count_nonzero(mask.array), args.winspec)

    logging.info('Calculating %s texture features for %s...', args.method,
                 args.mode)

    dwi.rcParams.texture_avg = args.voxel
    if dwi.rcParams.texture_avg != 'all':
        if args.mode.startswith('T2w') and args.method.startswith('gabor'):
            # These result arrays can get quite huge (if float64).
            dwi.rcParams.texture_path = args.output

    if args.method in ('glcm', 'glcm_mbb'):
        img = dwi.util.quantize(dwi.util.normalize(img, args.mode))

    tmap, names = dwi.texture.get_texture(img, args.method, args.winspec,
                                          pmask)
    attrs['parameters'] = names
    # Number of windows, or resulting texture map volume in general.
    attrs['tmap_voxels'] = np.count_nonzero(pmask)

    logging.info('Writing shape %s, type %s to %s', tmap.shape, tmap.dtype,
                 args.output)
    if dwi.rcParams.texture_path:
        attrs['shape'] = tmap.shape
        attrs['dtype'] = str(tmap.dtype)
        dwi.hdf5.write_attrs(tmap, attrs)  # Attributes may need conversion.
    else:
        dwi.files.write_pmap(args.output, tmap, attrs)


if __name__ == '__main__':
    main()
