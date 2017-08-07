#!/usr/bin/python3

"""Get grid-wise features."""

# TODO Also scale lesiontype.

import argparse
from itertools import product
import logging
import os.path

import numpy as np
from scipy import ndimage

import dwi.files
import dwi.texture
import dwi.util

log = logging.getLogger('grid')


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
    p.add_argument('--mbb', type=float,
                   help='minimum bounding box padding in millimeters (try 15)')
    p.add_argument('--voxelsize', type=float,
                   help='rescaled voxel size in millimeters (try 0.25)')
    p.add_argument('--winsize', type=float, default=5,
                   help='window (cube) size in millimeters (default 5)')
    p.add_argument('--voxelspacing', type=float, nargs=3,
                   help='force voxel spacing (leave out to read from image)')
    p.add_argument('--use_centroid', action='store_true',
                   help='align by prostate centroid instead of image corner')
    p.add_argument('--nanbg', action='store_true',
                   help='set non-prostate background to nan')
    p.add_argument('--lesiontypes', metavar='TYPE', nargs='+',
                   help='lesion types in mask order (CZ or PZ)')
    p.add_argument('--output', metavar='FILENAME', required=True,
                   help='output pmap file')
    return p.parse_args()


def get_lesiontype_array(lesiontypes, lesions):
    """Create lesiontype array. It contains -1 or 1 depending on lesion type,
    or zero where no lesion.
    """
    lesiontype = np.zeros_like(lesions[0], dtype=np.int8)
    if lesiontypes is not None:
        for lt, l in zip(lesiontypes, lesions):
            if lt.lower() == 'cz':
                lesiontype[l] = -1
            elif lt.lower() == 'pz':
                lesiontype[l] = 1
            else:
                raise ValueError('Invalid lesiontype: {}'.format(lt))
    log.info('Lesion types: %s, +1: %i, -1: %i', lesiontypes,
             np.count_nonzero(lesiontype == 1),
             np.count_nonzero(lesiontype == -1))
    return lesiontype


def get_mbb(mask, spacing, pad):
    """Get mask minimum bounding box as slices, with minimum padding in mm."""
    padding = [int(np.ceil(pad / x)) for x in spacing]
    physical_padding = [x * y for x, y in zip(padding, spacing)]
    mbb = dwi.util.bounding_box(mask, padding)
    slices = tuple(slice(*x) for x in mbb)
    log.info('Cropping minimum bounding box, padding: %s', pad)
    log.debug('\tVoxel padding: %s', padding)
    log.debug('\tPhysical padding: %s', physical_padding)
    log.debug('\tMinimum bounding box: %s', mbb)
    return slices


def rescale(img, src_spacing, dst_spacing):
    """Rescale image according to voxel spacing sequences (mm per voxel)."""
    factor = [s/d for s, d in zip(src_spacing, dst_spacing)]
    log.info('Scaling by factor: %s', factor)
    output = ndimage.interpolation.zoom(img, factor, order=0)
    return output


def generate_windows(imageshape, winshape, center):
    """Generate slice objects for a grid of windows around given center.

    Float center will be rounded. Yield a tuple with coordinate slices of each
    window, and window position relative to the center.
    """
    center = [int(round(x)) for x in center]
    starts = [i % w for i, w in zip(center, winshape)]
    stops = [i-w+1 for i, w in zip(imageshape, winshape)]
    its = (range(*x) for x in zip(starts, stops, winshape))
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


def create_grid_centroid(metric_winshape, metric_gridshape=(100, 150, 150)):
    """Create and fill grid array based on prostate centroid."""
    gridshape = [int(g//w) for g, w in zip(metric_gridshape, metric_winshape)]
    gridshape = [x + x % 2 for x in gridshape]  # Make any odds even.
    grid = np.full(gridshape + [4], np.nan, dtype=np.float32)
    return grid


def create_grid_corner(image, winshape):
    """Create and fill grid array based on corner."""
    gridshape = [i//w for i, w in zip(image.shape, winshape)]
    grid = np.full(gridshape + [4], np.nan, dtype=np.float32)
    return grid


def process(image, spacing, prostate, lesion, lesiontype, metric_winshape,
            stat, voxelsize=None, use_centroid=False):
    """Process one parameter."""
    # TODO: Should do them all at the same time.
    # Rescale image and masks.
    if voxelsize is not None:
        src_spacing = spacing
        spacing = [voxelsize] * 3
        image = rescale(image, src_spacing, spacing)
        prostate = prostate.astype(np.float_)
        prostate = rescale(prostate, src_spacing, spacing)
        prostate = dwi.util.asbool(prostate)
        lesion = lesion.astype(np.float_)
        lesion = rescale(lesion, src_spacing, spacing)
        lesion = dwi.util.asbool(lesion)
        assert image.shape == prostate.shape == lesion.shape
        # TODO Also scale lesiontype.

        phys_size = [x*y for x, y in zip(image.shape, spacing)]
        log.info('Transformed image: %s %s', image.shape, image.dtype)
        log.info('Voxel spacing: %s, physical size: %s', spacing, phys_size)

    # Extract grid datapoints. Grid placing is based either on prostate
    # centroid, or image corner.
    voxel_winshape = [int(round(x/y)) for x, y in zip(metric_winshape,
                                                      spacing)]
    log.debug('Window metric: %s, voxel: %s', metric_winshape, voxel_winshape)

    centroid = [round(x, 2) for x in dwi.util.centroid(prostate)]
    if use_centroid:
        base = centroid
        grid = create_grid_centroid(metric_winshape)
        grid_base = [s//2 for s in grid.shape]
    else:
        base = [0] * 3
        grid = create_grid_corner(image, voxel_winshape)
        grid_base = [0] * 3
    log.debug('Prostate centroid: %s, base: %s', centroid, base)

    windows = list(generate_windows(image.shape, voxel_winshape, base))
    for slices, relative in windows:
        indices = tuple(c+r for c, r in zip(grid_base, relative))
        values = get_datapoint(image[slices], prostate[slices], lesion[slices],
                               lesiontype[slices], stat)
        grid[indices] = values
    return grid


def average_image(image):
    """Do average filtering for image."""
    d = dict(size=(3, 3), mode='nearest')
    for p in range(image.shape[-1]):
        for i in range(image.shape[0]):
            ix = (i, slice(None), slice(None), p)
            image[ix] = ndimage.filters.median_filter(image[ix], **d)


def indexed_path(path, i):
    """Add an index to path before possible extension."""
    root, ext = os.path.splitext(path)
    return '{r}-{i}{e}'.format(r=root, i=i, e=ext)


def set_loggin(verbosity=0):
    """Set up logging."""
    import sys
    loglevel = logging.INFO if verbosity else logging.WARNING
    logging.basicConfig(level=loglevel, stream=sys.stdout)


def main():
    """Main."""
    args = parse_args()
    set_loggin(verbosity=args.verbose)

    image, attrs = dwi.files.read_pmap(args.image, ondisk=True)
    if args.param is not None:
        image = image[..., args.param]
        image.shape += (1,)
        attrs['parameters'] = [attrs['parameters'][args.param]]
    spacing = attrs['voxel_spacing']

    # Read masks.
    prostate = dwi.files.read_mask(args.prostate,
                                   expected_voxel_spacing=spacing)
    lesions = [dwi.files.read_mask(x, expected_voxel_spacing=spacing,
                                   container=prostate) for x in args.lesions]
    lesion = dwi.util.unify_masks(lesions)

    assert image.shape[:3] == prostate.shape == lesion.shape
    if args.voxelspacing is not None:
        spacing = args.voxelspacing

    phys_size = [x*y for x, y in zip(image.shape[:3], spacing)]
    log.info('Image: %s %s', image.shape, image.dtype)
    log.debug('Voxel spacing: %s, physical size: %s', spacing, phys_size)
    log.debug('Lesions: %i', len(args.lesions))

    lesiontype = get_lesiontype_array(args.lesiontypes, lesions)

    # Crop MBB. The remaining image is stored in memory.
    if args.mbb is None:
        slices = tuple(slice(0, x) for x in image.shape[:3])
    else:
        slices = get_mbb(prostate, spacing, args.mbb)
    image = image[slices]
    prostate = prostate[slices]
    lesion = lesion[slices]
    lesiontype = lesiontype[slices]
    assert (image.shape[:3] == prostate.shape == lesion.shape ==
            lesiontype.shape)

    # average_image(image)

    assert image.ndim == 4, image.ndim
    image = image.astype(np.float32)
    if args.nanbg:
        image[-prostate] = np.nan  # Set background to nan.

    basic = ['prostate', 'lesion', 'lesiontype']
    metric_winshape = [args.winsize] * 3
    if args.param is None:
        params = attrs['parameters']  # Use average of each parameter.
    else:
        params = list(dwi.texture.stats([0]).keys())  # Use statistical feats.
    d = dict(voxelsize=args.voxelsize, use_centroid=args.use_centroid)
    grid = None
    for i, param in enumerate(params):
        if args.param is None:
            img = image[..., i]
            stat = None
        else:
            img = image[..., 0]
            stat = param
        a = process(img, spacing, prostate, lesion, lesiontype,
                    metric_winshape, stat, **d)
        if grid is None:
            shape = a.shape[0:-1] + (len(basic) + len(params),)
            grid = np.empty(shape, dtype=a.dtype)
            log.info('Grid shape: %s', grid.shape)
            grid[..., 0:len(basic)] = a[..., 0:-1]  # Init with basic.
        grid[..., len(basic)+i] = a[..., -1]  # Add each feature.
    outfile = args.output
    attrs = dict(n_lesions=len(args.lesions), spacing=metric_winshape)
    attrs['parameters'] = basic + params
    log.info('Writing %s to %s', grid.shape, outfile)
    dwi.files.write_pmap(outfile, grid, attrs)


if __name__ == '__main__':
    main()
