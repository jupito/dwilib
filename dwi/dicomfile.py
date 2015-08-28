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
    positions = set()
    bvalues = set()
    slices = dict()  # Lists of single slices indexed by (position, bvalue).
    d = {}
    for f in filenames:
        df = dicom.read_file(f)
        if 'PixelData' not in df:
            continue
        d.setdefault('orientation', df.ImageOrientationPatient)
        if d['orientation'] != df.ImageOrientationPatient:
            raise Exception('Orientation mismatch.')
        d.setdefault('shape', df.pixel_array.shape)
        if d['shape'] != df.pixel_array.shape:
            raise Exception('Shape mismatch.')
        d.setdefault('type', df.pixel_array.dtype)
        if d['type'] != df.pixel_array.dtype:
            raise Exception('Type mismatch.')
        d.setdefault('voxel_spacing', get_voxel_spacing(df))
        position = tuple(float(x) for x in df.ImagePositionPatient)
        bvalue = get_bvalue(df)
        pixels = get_pixels(df)
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
    r = dict(bvalues=bvalues, voxel_spacing=d['voxel_spacing'], image=image)
    return r


def construct_image(slices, positions, bvalues):
    """Construct uniform image array from slice dictionary."""
    w, h = slices.values()[0].shape
    shape = (len(positions), w, h, len(bvalues))
    image = np.empty(shape, dtype=np.float32)
    image.fill(np.nan)
    for k, v in slices.iteritems():
        i = positions.index(k[0])
        j = bvalues.index(k[1])
        image[i, :, :, j] = v
    if np.isnan(np.min(image)):
        raise Exception('Slices missing from shape {:s}.'.format(shape))
    return image


def get_bvalue(df):
    """Return image b-value. It may also be stored as frame second."""
    if 'DiffusionBValue' in df:
        r = df.DiffusionBValue
    elif 'FrameTime' in df:
        r = df.FrameTime / 1000
    elif 'FrameReferenceTime' in df:
        r = df.FrameReferenceTime / 1000
    else:
        raise AttributeError('DICOM file does not contain a b-value')
    return r


def get_pixels(df):
    """Return rescaled pixel array from DICOM object."""
    pixels = df.pixel_array.astype(np.float32)
    pixels = pixels * df.RescaleSlope + df.RescaleIntercept
    # # Clipping should not be done.
    # lowest = df.WindowCenter - df.WindowWidth/2
    # highest = df.WindowCenter + df.WindowWidth/2
    # pixels = pixels.clip(lowest, highest, out=pixels)
    return pixels


def get_voxel_spacing(df):
    """Return voxel spacing in millimeters as (z, y, x)."""
    # Note: Some manufacturers misinterpret SpacingBetweenSlices, it would be
    # better to calculate this from ImageOrientationPatient and
    # ImagePositionPatient.
    z = df.SpacingBetweenSlices if 'SpacingBetweenSlices' in df else 1.
    x, y = df.PixelSpacing if 'PixelSpacing' in df else (1., 1.)
    return tuple(float(n) for n in (z, y, x))
