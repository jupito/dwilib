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
    d = {}
    for f in filenames:
        read_slice(f, d)
    positions = sorted(d['positions'])
    bvalues = sorted(d['bvalues'])
    slices = d['slices']
    # If any slices are scanned multiple times, use mean.
    for k, v in slices.iteritems():
        slices[k] = np.mean(v, axis=0)
    image = construct_image(slices, positions, bvalues)
    r = dict(bvalues=bvalues, voxel_spacing=d['voxel_spacing'], image=image)
    return r


def read_slice(filename, d):
    """Read a single slice."""
    df = dicom.read_file(filename)
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
    pixels = get_pixels(df)
    d.setdefault('positions', set()).add(position)
    d.setdefault('bvalues', set()).add(bvalue)
    key = (position, bvalue)
    slices = d.setdefault('slices', {})  # Indexed by (position, bvalue)...
    slices.setdefault(key, []).append(pixels)  # ...are lists of slices.


def read_files_(filenames):
    """Read a bunch of files, each containing a single slice with one b-value,
    and construct a 4d image array.

    The slices are sorted simply by their position as it is, assuming it only
    changes in one dimension. In case there are more than one scan of
    a position and a b-value, the files are averaged by mean.

    DICOM files without pixel data are silently skipped.
    """
    slicedicts = [read_slice_(x) for x in filenames]
    slicedicts = [x for x in slicedicts if x is not None]
    r = {k: slicedicts[0][k] for k in ['dicom_shape', 'dicom_type',
                                       'orientation', 'voxel_spacing']}
    for k, v in r.iteritems():
        # These must match in all slice files.
        if any(x[k] != v for x in slicedicts):
            raise ValueError('DICOM header mismatch: {}'.format(k))
    positions = sorted(set(x['position'] for x in slicedicts))
    bvalues = sorted(set(x['bvalue'] for x in slicedicts))
    slices = {}
    for d in slicedicts:
        key = (d['position'], d['bvalue'])  # Indexed by (position, bvalue)...
        slices.setdefault(key, []).append(d['pixels'])  # ...are slice lists.
    # If any slices are scanned multiple times, use mean.
    for k, v in slices.iteritems():
        slices[k] = np.mean(v, axis=0)
    image = construct_image(slices, positions, bvalues)
    r.update(image=image, bvalues=bvalues, positions=positions)
    return r


def read_slice_(filename):
    """Read a single slice."""
    df = dicom.read_file(filename)
    if 'PixelData' not in df:
        return None
    d = dict(
        dicom_shape=df.pixel_array.shape,
        dicom_type=df.pixel_array.dtype,
        orientation=df.ImageOrientationPatient,
        voxel_spacing=get_voxel_spacing(df),
        position=tuple(float(x) for x in df.ImagePositionPatient),
        bvalue=get_bvalue(df),
        pixels=get_pixels(df),
        )
    return d


def construct_image(slices, positions, bvalues):
    """Construct uniform image array from slice dictionary."""
    w, h = slices.values()[0].shape
    dtype = slices.values()[0].dtype
    shape = (len(positions), w, h, len(bvalues))
    image = np.empty(shape, dtype=dtype)
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
