"""Support for reading DWI data from DICOM files."""

from __future__ import absolute_import, division, print_function
import os.path

import numpy as np

import dicom


def read_dir(dirname):
    """Read a directory containing DICOM files. See dicomfile.read_files().
    """
    # Sometimes the files reside in an additional 'DICOM' subdirectory.
    path = os.path.join(dirname, 'DICOM')
    if os.path.isdir(path):
        dirname = path
    filenames = os.listdir(dirname)
    pathnames = [os.path.join(dirname, f) for f in filenames]
    return read_files(pathnames)


def read_files(filenames):
    """Read a bunch of files, each containing a single slice with one b-value,
    and construct a 4d image array.

    The slices are sorted simply by their position as it is, assuming it only
    changes in one dimension. In case there are more than one scan of
    a position and a b-value, the files are averaged by mean.

    DICOM files without pixel data are silently skipped.
    """
    orientation = None
    shape = None
    voxel_spacing = None
    positions = set()
    bvalues = set()
    slices = dict()  # Lists of single slices indexed by (position, bvalue).
    for f in filenames:
        d = dicom.read_file(f)
        if 'PixelData' not in d:
            continue
        orientation = orientation or d.ImageOrientationPatient
        if d.ImageOrientationPatient != orientation:
            raise Exception('Orientation mismatch.')
        shape = shape or d.pixel_array.shape
        if d.pixel_array.shape != shape:
            raise Exception('Shape mismatch.')
        voxel_spacing = voxel_spacing or get_voxel_spacing(d)
        position = tuple(map(float, d.ImagePositionPatient))
        bvalue = get_bvalue(d)
        pixels = get_pixels(d)
        positions.add(position)
        bvalues.add(bvalue)
        key = (position, bvalue)
        slices.setdefault(key, []).append(pixels)
    positions = sorted(positions)
    bvalues = sorted(bvalues)
    # If any slices are scanned multiple times, use mean.
    for k, v in slices.iteritems():
        slices[k] = np.mean(v, axis=0)
    image = construct_image(slices, positions, bvalues)
    d = dict(bvalues=bvalues, voxel_spacing=voxel_spacing, image=image)
    return d


def construct_image(slices, positions, bvalues):
    """Construct uniform image array from slice dictionary."""
    w, h = slices.values()[0].shape
    shape = (len(positions), w, h, len(bvalues))
    image = np.empty(shape)
    image.fill(np.nan)
    for k, v in slices.iteritems():
        i = positions.index(k[0])
        j = bvalues.index(k[1])
        image[i, :, :, j] = v
    if np.isnan(np.min(image)):
        raise Exception('Slices missing from shape {:s}.'.format(shape))
    return image


def get_bvalue(d):
    """Return image b-value. It may also be stored as frame second."""
    if 'DiffusionBValue' in d:
        r = d.DiffusionBValue
    elif 'FrameTime' in d:
        r = d.FrameTime / 1000
    elif 'FrameReferenceTime' in d:
        r = d.FrameReferenceTime / 1000
    else:
        raise AttributeError('DICOM file does not contain a b-value')
    return r


def get_pixels(d):
    """Return rescaled pixel array from DICOM object."""
    pixels = d.pixel_array.astype(float)  # XXX: How about float32?
    pixels = pixels * d.RescaleSlope + d.RescaleIntercept
    # # Clipping should not be done.
    # lowest = d.WindowCenter - d.WindowWidth/2
    # highest = d.WindowCenter + d.WindowWidth/2
    # pixels = pixels.clip(lowest, highest, out=pixels)
    return pixels


def get_voxel_spacing(d):
    """Return voxel spacing in millimeters as (z, y, x)."""
    # Note: Some manufacturers misinterpret SpacingBetweenSlices, it would be
    # better to calculate this from ImageOrientationPatient and
    # ImagePositionPatient.
    z = d.SpacingBetweenSlices if 'SpacingBetweenSlices' in d else 1.
    x, y = d.PixelSpacing if 'PixelSpacing' in d else (1., 1.)
    return tuple(map(float, (z, y, x)))
