"""PyDoIt common stuff."""

# TODO: Merge cases_scans() and lesions(). Remove read_sample_list().

from itertools import chain, product

from doit.tools import create_folder

from . import files, util
from . import rcParams
from .paths import samplelist_path
from .types import Path, TextureSpec

# # Imaging modes.
# DEFAULT_MODE = 'DWI-Mono-ADCm'
# MODES = [ImageMode(x) for x in words(get_var('mode', DEFAULT_MODE))]
# # Sample lists (train, test, etc).
# SAMPLELISTS = words(get_var('samplelist', 'all'))


def get_config():
    """Get doit config (DOIT_CONFIG)."""
    return {
        # 'backend': 'sqlite3',
        'default_tasks': [],
        'verbosity': 1,
        # 'num_process': 7,
        'num_process': get_num_process(),
    }


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
    return int(max(1, n))


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
                yield p.num, scan, i + 1


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
            yield TextureSpec(method, winsize, None)


def texture_params(voxels=None):
    """Iterate texture parameter combinations."""
    masktypes = ['lesion']
    slices = ['maxfirst', 'all']
    portion = [1, 0]
    voxels = iter(voxels or ['mean', 'median', 'all'])
    return product(masktypes, slices, portion, voxels)


def find_roi_param_combinations(mode, samplelist):
    """Generate all find_roi.py parameter combinations."""
    find_roi_params = [
        [1, 2, 3],  # ROI depth min
        [1, 2, 3],  # ROI depth max
        range(2, 13),  # ROI side min (3 was not good)
        range(3, 13),  # ROI side max
        chain(range(250, 2000, 250), [50, 100, 150, 200]),  # Number of ROIs
    ]
    if mode[0] == 'DWI':
        if samplelist == 'test':
            params = [
                (2, 3, 10, 10, 500),  # Mono: corr, auc
                (2, 3, 10, 10, 1750),  # Mono: corr
                (2, 3, 11, 11, 750),  # Mono: corr
                # (2, 3, 2, 2, 250),  # Kurt: auc
                # (2, 3, 9, 9, 1000),  # Kurt: corr
                # (2, 3, 12, 12, 1750),  # Kurt: corr
                # (2, 3, 5, 5, 500),  # Kurt K: corr, auc
            ]
        else:
            params = product(*find_roi_params)
        for t in params:
            if t[0] <= t[1] and t[2] == t[3]:
                yield [str(x) for x in t]
