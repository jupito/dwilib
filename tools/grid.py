#!/usr/bin/env python2

"""Get grid-wise features."""

from __future__ import absolute_import, division, print_function
import argparse
from itertools import product
import os.path

import numpy as np
import scipy.ndimage
import scipy.stats

import dwi.files
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
    p.add_argument('--output', metavar='FILENAME', required=True,
                   help='output ASCII file')
    return p.parse_args()


def read_mask(path, expected_voxel_spacing, n_dec=3, container=None,
              allowed_outside=0.2):
    """Read pmap as a mask. Expect voxel spacing to match up to a certain
    number of decimals.

    The optional parameter allowed_outside indicates how much of the lesion
    volume may be outside of the prostate without an error being raised.
    """
    mask, attrs = dwi.files.read_pmap(path)
    mask = mask[..., 0].astype(np.bool)
    voxel_spacing = [round(x, n_dec) for x in attrs['voxel_spacing']]
    expected_voxel_spacing = [round(x, n_dec) for x in expected_voxel_spacing]
    if voxel_spacing != expected_voxel_spacing:
        raise ValueError('Expected voxel spacing {}, got {}'.format(
            expected_voxel_spacing, voxel_spacing))
    if container is not None:
        portion_outside_container = (np.count_nonzero(mask[~container]) /
                                     np.count_nonzero(mask))
        if portion_outside_container > allowed_outside:
            s = '{}: Portion of selected voxels outside container is {:%}'
            raise ValueError(s.format(path, portion_outside_container))
    return mask


def unify_masks(masks):
    """Unify a sequence of masks into one."""
    # return np.sum(masks, axis=0, dtype=np.bool)
    return reduce(np.maximum, masks)


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
    # print('Scaling, factor:', factor)
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


def get_datapoint(image, prostate, lesion):
    """Extract output datapoint for a cube.

    The cube window is included if at least half of it is of prostate.
    """
    assert image.shape == prostate.shape == lesion.shape
    if np.isnan(image).all():
        value = np.nan
    else:
        value = np.nanmean(image)
    return (
        np.count_nonzero(prostate) / prostate.size,
        np.count_nonzero(lesion) / prostate.size,
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


def filled(shape, value, **kwargs):
    """Return a new array of given shape and type, initialized by value."""
    a = np.empty(shape, **kwargs)
    a.fill(value)
    return a


def process(image, voxel_spacing, prostate, lesion, voxelsize,
            metric_winshape, verbose):
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

    # TODO: Output grid size is hard-coded here for now.
    a = filled((20, 30, 30, 3), np.nan, dtype=np.float32)
    for slices, relative in windows:
        indices = tuple(s/2+r for s, r in zip(a.shape, relative))
        values = get_datapoint(image[slices], prostate[slices], lesion[slices])
        a[indices] = values
    return a


def main():
    args = parse_args()
    image, attrs = dwi.files.read_pmap(args.image, ondisk=True)
    if args.param is not None:
        image = image[..., args.param]
        image.shape += (1,)
        attrs['parameters'] = [attrs['parameters'][args.param]]
    voxel_spacing = attrs['voxel_spacing']
    prostate = read_mask(args.prostate, voxel_spacing)
    lesion = unify_masks([read_mask(x, voxel_spacing, container=prostate) for
                          x in args.lesions])
    if args.verbose:
        print('Lesions:', len(args.lesions))
    assert image.shape[:3] == prostate.shape == lesion.shape

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
    assert image.shape[:3] == prostate.shape == lesion.shape

    assert image.ndim == 4, image.ndim
    image = image.astype(np.float32)
    image[-prostate] = np.nan  # XXX: Is it ok to set background as nan?

    metric_winshape = (args.winsize,) * 3
    root, ext = os.path.splitext(args.output)
    for i, param in enumerate(attrs['parameters']):
        a = process(image[..., i], voxel_spacing, prostate, lesion,
                    args.voxelsize, metric_winshape, args.verbose)
        if ext == '.txt':
            # Exclude non-prostate cubes from ASCII output.
            nans = np.isnan(a[..., -1])
            a = a[~nans]
        params = ['prostate', 'lesion', param]
        attrs = dict(parameters=params, n_lesions=len(args.lesions),
                     voxel_spacing=metric_winshape)
        outfile = '{r}-{i}{e}'.format(r=root, i=i, e=ext)
        if args.verbose:
            print('Writing to {}'.format(outfile))
        dwi.files.write_pmap(outfile, a, attrs)


if __name__ == '__main__':
    main()
