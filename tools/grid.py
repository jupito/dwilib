#!/usr/bin/env python2

"""Get grid-wise features."""

# TODO Also scale lesiontype.

from __future__ import absolute_import, division, print_function
import argparse
from itertools import product
import os.path

import numpy as np
import scipy.ndimage
import scipy.stats

import dwi.files
import dwi.texture
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--image', required=True,
                   help='input image or pmap')
    p.add_argument('--param', type=int,
                   help='parameter index')
    p.add_argument('--prostate', metavar='MASKFILE', required=True,
                   help='prostate mask')
    p.add_argument('--lesions', metavar='MASKFILE', nargs='+', required=True,
                   help='lesion masks')
    p.add_argument('--mbb', type=float, required=True,
                   help='minimum bounding box padding in millimeters')
    p.add_argument('--voxelsize', type=float,
                   help='rescaled voxel size in millimeters (try 0.25)')
    p.add_argument('--winsize', type=float, default=5,
                   help='window (cube) size in millimeters (default 5)')
    p.add_argument('--voxelspacing', type=float, nargs=3,
                   help='force voxel spacing (leave out to read from image)')
    p.add_argument('--lesiontypes', metavar='TYPE', nargs='+',
                   help='lesion types in mask order (CZ or PZ)')
    p.add_argument('--output', metavar='FILENAME', required=True,
                   help='output ASCII file')
    return p.parse_args()


def get_mbb(mask, voxel_spacing, pad):
    """Get mask minimum bounding box as slices, with minimum padding in mm."""
    padding = tuple(int(np.ceil(pad / x)) for x in voxel_spacing)
    physical_padding = tuple(x * y for x, y in zip(padding, voxel_spacing))
    mbb = dwi.util.bounding_box(mask, padding)
    slices = tuple(slice(*x) for x in mbb)
    print('Cropping minimum bounding box with pad:', pad)
    print('\tVoxel padding:', padding)
    print('\tPhysical padding:', physical_padding)
    print('\tMinimum bounding box:', mbb)
    return slices


def rescale(img, src_voxel_spacing, dst_voxel_spacing):
    """Rescale image according to voxel spacing sequences (mm per voxel)."""
    factor = tuple(s/d for s, d in zip(src_voxel_spacing, dst_voxel_spacing))
    print('Scaling, factor:', factor)
    output = scipy.ndimage.interpolation.zoom(img, factor, order=0)
    return output


def float2bool_mask(a):
    """Convert float array to boolean mask (round, clip to [0, 1])."""
    a = a.round()
    a.clip(0, 1, out=a)
    a = a.astype(np.bool)
    return a


def generate_windows(imageshape, winshape, center):
    """Generate slice objects for a grid of windows around given center.

    Float center will be rounded. Yield a tuple with coordinate slices of each
    window, and window position relative to the center.
    """
    center = [int(round(x)) for x in center]
    starts = [i % w for i, w in zip(center, winshape)]
    stops = [i-w+1 for i, w in zip(imageshape, winshape)]
    its = (xrange(*x) for x in zip(starts, stops, winshape))
    for coords in product(*its):
        slices = tuple(slice(i, i+w) for i, w in zip(coords, winshape))
        relative = tuple(int((i-c)/w) for i, c, w in zip(coords, center,
                                                         winshape))
        yield slices, relative


def get_datapoint(image, prostate, lesion, lesiontype, stat):
    """Extract output datapoint for a cube.

    If stat is None, median is used. Otherwise, see dwi.texture.stats().
    """
    assert image.shape == prostate.shape == lesion.shape == lesiontype.shape
    if np.isnan(image).all():
        value = np.nan
    else:
        image = image[np.isfinite(image)]  # Remove nan values.
        if stat is None:
            value = np.median(image)
        else:
            value = dwi.texture.stats(image)[stat]
    nneg = np.count_nonzero(lesiontype < 0)
    npos = np.count_nonzero(lesiontype > 0)
    # Label as lesiontype -1 or 1 based on majority, or 0 if no lesion.
    lt = 0
    if nneg > 0:
        lt = -1
    if npos > nneg:
        lt = 1
    return (
        np.count_nonzero(prostate) / prostate.size,
        np.count_nonzero(lesion) / prostate.size,
        lt,
        value,
    )


def print_correlations(data, params):
    """Print correlations for testing."""
    data = np.asarray(data)
    print(data.shape, data.dtype)
    indices = range(data.shape[-1])
    for i, j in product(indices, indices):
        if i < j:
            rho, pvalue = scipy.stats.spearmanr(data[:, i], data[:, j])
            s = 'Spearman: {:8} {:8} {:+1.4f} {:+1.4f}'
            print(s.format(params[i], params[j], rho, pvalue))


def process(image, voxel_spacing, prostate, lesion, lesiontype, voxelsize,
            metric_winshape, verbose, stat):
    """Process one parameter."""
    # Rescale image and masks.
    if voxelsize is not None:
        src_voxel_spacing = voxel_spacing
        voxel_spacing = (voxelsize,) * 3
        image = rescale(image, src_voxel_spacing, voxel_spacing)
        prostate = prostate.astype(np.float_)
        prostate = rescale(prostate, src_voxel_spacing, voxel_spacing)
        prostate = float2bool_mask(prostate)
        lesion = lesion.astype(np.float_)
        lesion = rescale(lesion, src_voxel_spacing, voxel_spacing)
        lesion = float2bool_mask(lesion)
        assert image.shape == prostate.shape == lesion.shape
        # TODO Also scale lesiontype.

    if verbose:
        physical_size = tuple(x*y for x, y in zip(image.shape, voxel_spacing))
        print('Transformed image:', image.shape, image.dtype)
        print('\tVoxel spacing:', voxel_spacing)
        print('\tPhysical size:', physical_size)

    # Extract grid datapoints.
    voxel_winshape = tuple(int(round(x/y)) for x, y in zip(metric_winshape,
                                                           voxel_spacing))
    centroid = dwi.util.centroid(prostate)
    if verbose:
        print('Window shape (metric, voxel):', metric_winshape, voxel_winshape)
        print('Prostate centroid:', centroid)
    windows = list(generate_windows(image.shape, voxel_winshape, centroid))

    # Create and fill grid array.
    # TODO: Should determine output grid size from prostate size.
    gridshape = (20, 30, 30, 4)
    grid = np.full(gridshape, np.nan, dtype=np.float32)
    for slices, relative in windows:
        indices = tuple(s/2+r for s, r in zip(grid.shape, relative))
        values = get_datapoint(image[slices], prostate[slices], lesion[slices],
                               lesiontype[slices], stat)
        grid[indices] = values
    return grid


def main():
    args = parse_args()
    image, attrs = dwi.files.read_pmap(args.image, ondisk=True)
    if args.param is not None:
        image = image[..., args.param]
        image.shape += (1,)
        attrs['parameters'] = [attrs['parameters'][args.param]]
    voxel_spacing = attrs['voxel_spacing']
    prostate = dwi.files.read_mask(args.prostate,
                                   expected_voxel_spacing=voxel_spacing)
    lesions = [dwi.files.read_mask(x, expected_voxel_spacing=voxel_spacing,
                                   container=prostate) for x in args.lesions]
    lesion = dwi.util.unify_masks(lesions)
    lesiontype = np.zeros_like(lesion, dtype=np.int8)
    if args.verbose:
        print('Lesions:', len(args.lesions))
    assert image.shape[:3] == prostate.shape == lesion.shape
    if args.voxelspacing is not None:
        voxel_spacing = args.voxelspacing
    if args.lesiontypes is not None:
        print('Lesion types:', args.lesiontypes)
        for lt, l in zip(args.lesiontypes, lesions):
            if lt.lower() == 'cz':
                lesiontype[l] = -1
            elif lt.lower() == 'pz':
                lesiontype[l] = 1
            else:
                raise ValueError('Invalid lesiontype: {}'.format(lt))
    print(lesiontype.mean(), np.count_nonzero(lesiontype == 1),
          np.count_nonzero(lesiontype == -1))

    if args.verbose:
        physical_size = tuple(x*y for x, y in zip(image.shape[:3],
                                                  voxel_spacing))
        print('Image:', image.shape, image.dtype)
        print('\tVoxel spacing:', voxel_spacing)
        print('\tPhysical size:', physical_size)

    # Crop MBB. The remaining image is now stored in memory.
    slices = get_mbb(prostate, voxel_spacing, args.mbb)
    image = image[slices]
    prostate = prostate[slices]
    lesion = lesion[slices]
    lesiontype = lesiontype[slices]
    assert (image.shape[:3] == prostate.shape == lesion.shape ==
            lesiontype.shape)

    assert image.ndim == 4, image.ndim
    image = image.astype(np.float32)
    image[-prostate] = np.nan  # Set background to nan.

    metric_winshape = (args.winsize,) * 3
    root, ext = os.path.splitext(args.output)
    for i, param in enumerate(attrs['parameters']):
        a = process(image[..., i], voxel_spacing, prostate, lesion, lesiontype,
                    args.voxelsize, metric_winshape, args.verbose, None)
        if ext == '.txt':
            # Exclude non-prostate cubes from ASCII output.
            nans = np.isnan(a[..., -1])
            a = a[~nans]
        params = ['prostate', 'lesion', 'lesiontype', param]
        attrs = dict(parameters=params, n_lesions=len(args.lesions),
                     voxel_spacing=metric_winshape)
        outfile = '{r}-{i}{e}'.format(r=root, i=i, e=ext)
        if args.verbose:
            print('Writing to {}'.format(outfile))
        dwi.files.write_pmap(outfile, a, attrs)


if __name__ == '__main__':
    main()
