"""Support for reading DWI data from DICOM files."""

from __future__ import absolute_import, division, print_function
import os.path

import numpy as np
import dicom

import dwi.util


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
    d = {}
    for f in filenames:
        read_slice(f, d)
    positions = sorted(d['positions'])
    bvalues = sorted(d['bvalues'])
    echotimes = sorted(d['echotimes'])
    image = construct_image(d['slices'], positions, bvalues, echotimes)
    if len(bvalues) == image.shape[-1]:
        parameters = bvalues
    elif len(echotimes) == image.shape[-1]:
        parameters = echotimes
    else:
        raise ValueError('Inconsistent parameters')
    parameters = [str(x) for x in parameters]
    if len(positions) > 1:
        slice_spacing = get_slice_spacing(positions[0], positions[1])
        d['voxel_spacing'] = (slice_spacing,) + d['voxel_spacing'][1:]
    r = dict(image=image, bset=bvalues, echotimes=echotimes,
             parameters=parameters, voxel_spacing=d['voxel_spacing'])
    return r


def read_slice(filename, d):
    """Read a single slice."""
    try:
        df = dicom.read_file(filename)
    except dicom.filereader.InvalidDicomError as e:
        dwi.util.report('Error reading {f}: {e}'.format(f=filename, e=e))
        return
    if 'PixelData' not in df:
        return
    d.setdefault('orientation', df.ImageOrientationPatient)
    if d['orientation'] != df.ImageOrientationPatient:
        raise Exception('Orientation mismatch.')
    d.setdefault('shape', df.pixel_array.shape)
    if d['shape'] != df.pixel_array.shape:
        raise Exception('Shape mismatch.')
    d.setdefault('dtype', df.pixel_array.dtype)
    if d['dtype'] != df.pixel_array.dtype:
        raise Exception('Type mismatch.')
    d.setdefault('voxel_spacing', get_voxel_spacing(df))
    position = tuple(float(x) for x in df.ImagePositionPatient)
    bvalue = get_bvalue(df)
    echotime = get_echotime(df)
    pixels = get_pixels(df)
    d.setdefault('positions', set()).add(position)
    d.setdefault('bvalues', set()).add(bvalue)
    d.setdefault('echotimes', set()).add(echotime)
    key = (position, bvalue, echotime)
    slices = d.setdefault('slices', {})
    if key in slices:
        raise ValueError('Overlapping slices: {}, {}'.format(key, filename))
    slices[key] = pixels


def construct_image(slices, positions, bvalues, echotimes):
    """Construct uniform image array from slice dictionary."""
    w, h = slices.values()[0].shape
    dtype = slices.values()[0].dtype
    shape = (len(positions), w, h, len(bvalues), len(echotimes))
    image = np.empty(shape, dtype=dtype)
    image.fill(np.nan)
    for key, value in slices.iteritems():
        pos = positions.index(key[0])
        bv = bvalues.index(key[1])
        et = echotimes.index(key[2])
        image[pos, :, :, bv, et] = value
    if image.shape[3] == 1:
        image = image.squeeze(axis=3)
    elif image.shape[4] == 1:
        image = image.squeeze(axis=4)
    assert image.ndim == 4, image.shape
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
    if r is not None:
        r = int(r) if r.is_integer() else float(r)
    return r


def get_echotime(df):
    """Return Echo Time if present, or None."""
    r = df.get('EchoTime')
    if r is not None:
        r = int(r) if r.is_integer() else float(r)
    return r


def get_pixels(df):
    """Return rescaled pixel array from DICOM object."""
    pixels = df.pixel_array
    pixels = pixels.astype(np.float64)
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


def get_slice_spacing(pos1, pos2):
    """Calculate slice spacing by looking at the difference in two neighboring
    ImagePositionPatient sequences.
    """
    diffs = [abs(x-y) for x, y in zip(pos1, pos2)]
    diffs = [x for x in diffs if x != 0]
    if len(diffs) != 1:
        raise ValueError('Ambiguous slice spacing: {}, {}'.format(pos1, pos2))
    return diffs[0]
