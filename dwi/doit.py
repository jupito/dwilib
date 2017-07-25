"""PyDoIt common stuff."""

# TODO: Merge cases_scans() and lesions(). Remove read_sample_list().

from __future__ import absolute_import, division, print_function
import logging

from doit.tools import create_folder

from .paths import samplelist_path
from .types import Path, TextureSpec
from . import files, util
from . import rcParams


def get_num_process(factor=0.9, default=1):
    """Take a pick how many processes we want to run simultaneously."""
    maxjobs = rcParams.maxjobs
    try:
        if maxjobs < 0:
            # Joblib-type negative count: -1 => all, -2 => all but one, etc.
            n = util.cpu_count() + maxjobs + 1
        elif maxjobs < 1:
            # Portion of CPU count.
            n = util.cpu_count() * maxjobs
        else:
            # Absolute number.
            n = maxjobs
    except OSError:
        n = default
    n = int(max(1, n))
    logging.warning('Using %d processes', n)
    return n


def words(string, sep=','):
    """Split string into stripped words."""
    return [x.strip() for x in string.split(sep)]


def taskname(*items):
    """A task name consisting of items."""
    s = '_'.join('{}' for _ in items)
    return s.format(*items)


def folders(*paths):
    """A PyDoIt action that creates the folders for given file names """
    return [(create_folder, [str(Path(x).parent)]) for x in paths]


def _files(*paths):
    """Files for PyDoIt (It doesn't allow pathlib2, only str or pathlib)."""
    return [str(x) for x in paths]


def cases_scans(mode, samplelist):
    """Generate all case, scan pairs."""
    samples = files.read_sample_list(samplelist_path(mode, samplelist))
    for sample in samples:
        case = sample['case']
        for scan in sample['scans']:
            yield case, scan


def lesions(mode, samplelist):
    """Generate all case, scan, lesion# (1-based) combinations."""
    patients = files.read_patients_file(samplelist_path(mode, samplelist))
    for p in patients:
        for scan in p.scans:
            for i, _ in enumerate(p.lesions):
                yield p.num, scan, i+1


def texture_methods():
    """Return texture methods."""
    return rcParams.texture_methods


def texture_winsizes(masktype, mode, method):
    """Iterate texture window sizes."""
    if method == 'raw':
        return [1]
    elif method.endswith('_all'):
        return ['all']
    elif method.endswith('_mbb'):
        return ['mbb']
    elif method == 'sobel':
        return [3]  # Sobel convolution kernel is always 3x3 voxels.
    elif masktype in ('CA', 'N'):
        return [3, 5]  # These ROIs are always 5x5 voxels.
    elif mode[0] in ('T2', 'T2w'):
        return list(rcParams.texture_winsizes_large)
    else:
        return list(rcParams.texture_winsizes_small)


def texture_methods_winsizes(mode, masktype):
    """Generate texture method, window size combinations."""
    for method in texture_methods():
        for winsize in texture_winsizes(masktype, mode, method):
            yield TextureSpec(winsize, method, None)
