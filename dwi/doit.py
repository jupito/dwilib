"""PyDoIt common stuff."""

# TODO: Merge cases_scans() and lesions(). Remove read_sample_list().

from __future__ import absolute_import, division, print_function
from os.path import dirname

from doit.tools import create_folder

import dwi.files
from dwi.paths import samplelist_path


def get_num_process():
    """Take a pick how many processes we want to run simultaneously."""
    def get_hostname():
        """Return node hostname."""
        import platform
        return platform.uname()[1]
    d = dict(taanne=1, TY1303004=7, petmrc=24)
    return d[get_hostname()]


def words(string, sep=','):
    """Split string into stripped words."""
    return [x.strip() for x in string.split(sep)]


def name(*items):
    """A task name consisting of items."""
    s = '_'.join('{}' for _ in items)
    return s.format(*items)


def folders(*paths):
    """A PyDoIt action that creates the folders for given file names """
    return [(create_folder, [dirname(x)]) for x in paths]


def cases_scans(mode, samplelist):
    """Generate all case, scan pairs."""
    samples = dwi.files.read_sample_list(samplelist_path(mode, samplelist))
    for sample in samples:
        case = sample['case']
        for scan in sample['scans']:
            yield case, scan


def lesions(mode, samplelist):
    """Generate all case, scan, lesion# (1-based) combinations."""
    patients = dwi.files.read_patients_file(samplelist_path(mode, samplelist))
    for p in patients:
        for scan in p.scans:
            for i, _ in enumerate(p.lesions):
                yield p.num, scan, i+1


def texture_methods():
    return dwi.rcParams['texture.methods']


def texture_winsizes(masktype, mode, method):
    if method.endswith('_all'):
        return ['all']
    elif method.endswith('_mbb'):
        return ['mbb']
    elif method == 'sobel':
        return [3]  # Sobel convolution kernel is always 3x3 voxels.
    elif masktype in ('CA', 'N'):
        return [3, 5]  # These ROIs are always 5x5 voxels.
    elif mode[0] in ('T2', 'T2w'):
        return range(*dwi.rcParams['texture.winsizes.large'])
    else:
        return range(*dwi.rcParams['texture.winsizes.small'])


def texture_methods_winsizes(mode, masktype):
    for method in texture_methods():
        for winsize in texture_winsizes(masktype, mode, method):
            yield method, winsize
