"""Support for reading multi-slice DICOM images.

Use read() to read a directory, read_files() to read a group of files. Each
DICOM file may contain one slice, otherwise it will be ignored. Reading fails
unless all slices have equal orientation, shape, and pixel datatype. In case
any slices overlap, only one of them is kept. Between-slice spacing is
calculated from the first two slices' positional difference, as the
corresponding data field cannot be trusted.
"""

import logging
import re

import numpy as np
import pydicom

from .types import Path
from . import util

log = logging.getLogger(__name__)


def read(path):
    """Read a directory containing DICOM files. See dicomfile.read_files()."""
    path = Path(path)
    if path.is_file():
        return read_files([path])

    # If there's a single subdir, descend.
    entries = list(path.iterdir())
    if len(entries) == 1:
        entry, = entries
        if entry.is_dir():
            return read(entry)

    # Sometimes the files reside in an additional 'DICOM' subdirectory.
    entry = path / 'DICOM'
    if entry.is_dir():
        path = entry

    entries = list(path.iterdir())
    if not entries:
        raise ValueError('DICOM files not found: {}'.format(path))
    return read_files(entries)


def read_files(paths):
    """Read a bunch of files, each containing a single slice with one b-value,
    and construct a 4d image array.

    The slices are sorted simply by their position as it is, assuming it only
    changes in one dimension. In case there are more than one scan of
    a position and a b-value, the files are averaged by mean.

    DICOM files without pixel data are silently skipped.
    """
    d = dict(errors=[])
    for p in paths:
        read_slice(p, d)
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
    d['voxel_spacing'] = corrected_voxel_spacing(d['voxel_spacing'], positions)
    return dict(image=image, bset=bvalues, echotimes=echotimes,
                parameters=parameters, voxel_spacing=d['voxel_spacing'],
                dicom_dtype=str(d['dtype']), errors=d['errors'])


def read_slice(path, d):
    """Read a single slice."""
    try:
        df = pydicom.read_file(str(path))
    except pydicom.filereader.InvalidDicomError as e:
        log.error('Error reading %s: %s', path, e)
        return
    if 'PixelData' not in df:
        return
    d.setdefault('orientation', df.ImageOrientationPatient)
    if d['orientation'] != df.ImageOrientationPatient:
        raise Exception('Orientation mismatch.')
    d.setdefault('shape', df.pixel_array.shape)
    if d['shape'] != df.pixel_array.shape:
        raise Exception('Shape mismatch: {}'.format(path))
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
        log.error('Overlapping slices (%s), discarding %s', key, path)
        s = 'Overlapping slices, discarding {}'.format(path)
        d['errors'].append(s)
    slices[key] = pixels


def construct_image(slices, positions, bvalues, echotimes):
    """Construct uniform image array from slice dictionary."""
    slc = list(slices.values())[0]
    w, h = slc.shape
    dtype = slc.dtype
    shape = (len(positions), w, h, len(bvalues), len(echotimes))
    image = np.empty(shape, dtype=dtype)
    image.fill(np.nan)
    for key, value in slices.items():
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
    """Return image b-value. Default to 0 if not found.

    It may also be stored as frame second.

    Some Siemens scanners have it like this:
    (0018,0024) SH [*ep_b0]      #   6, 1 SequenceName
    (0018,0024) SH [*ep_b1500t]  #  10, 1 SequenceName
    """
    r = None
    if 'DiffusionBValue' in df:
        r = df.DiffusionBValue
    elif 'FrameTime' in df:
        r = df.FrameTime / 1000
    elif 'FrameReferenceTime' in df:
        r = df.FrameReferenceTime / 1000
    elif 'SequenceName' in df:
        s = df.SequenceName
        m = re.match(r'\*ep_b([0-9]+)t?', s)
        if m:
            r = int(m.group(1))
    if r is None:
        # log.warning('DICOM without b-value, defaulting to zero')
        r = 0
    if isinstance(r, float):
        # I'm not sure if they can be non-integer.
        r = int(r) if r.is_integer() else float(r)
    assert isinstance(r, int) or isinstance(r, float), r
    return r


def get_echotime(df):
    """Return Echo Time if present, or None."""
    r = df.get('EchoTime')
    if r is not None:
        r = int(r) if r.is_integer() else float(r)
    return r


def get_pixels(df, dtype=np.float32):
    """Return rescaled pixel array from DICOM object."""
    pixels = df.pixel_array
    pixels = pixels.astype(dtype)
    pixels = pixels * df.get('RescaleSlope', 1) + df.get('RescaleIntercept', 0)
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
    diffs = [abs(x - y) for x, y in zip(pos1, pos2)]
    if len([x for x in diffs if x > 0.05]) != 1:
        # More than one axis differs: use multi-axis distance.
        log.warning('Ambiguous slice spacing: %s, %s', pos1, pos2)
    return util.distance(pos1, pos2)


def corrected_voxel_spacing(voxel_spacing, positions):
    """Correct voxel spacing from slice positions, if possible."""
    if len(positions) > 1:
        slice_spacing = round(get_slice_spacing(positions[0], positions[1]), 5)
        return (slice_spacing,) + voxel_spacing[1:]
    return voxel_spacing
