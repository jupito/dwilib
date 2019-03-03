"""..."""

# TODO: Check what find_contours() params mean (high, high).
# TODO: Use masks as boolean, not float.
# TODO: Use skimage.draw.circle to draw a disk.
# TODO: Calculate histograms for prostate and lesion.
# TODO: Normalize images before blob detection?

# import contextlib
import csv
import logging
# import multiprocessing as mp
# from collections import OrderedDict
# from functools import partial
from itertools import chain
from pathlib import Path

import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import pandas as pd
# import scipy as sp
# from scipy import stats

import dwi.detectlesion
import dwi.files
import dwi.mask
import dwi.readnib
import dwi.stats
import dwi.util
from dwi.types2 import ImageMode, ImageTarget, GleasonScore
from dwi.util import one

PATIENT_INFO_PATH = dict(PRO3='patients_PRO3.tsv')
OUTDIR = Path('/home/jupito/Documents/work/roi_placement/tmp')
MODES = [ImageMode(*x) for x in [('DWI5b500', '', 'ADC'), ('T2w', '', 'T2W')]]
MODE = MODES[0]

log = logging.getLogger(__name__)


class PatientInfo:
    """Patient info read from the TSV file."""
    def __init__(self, info):
        self.info = dict(info)
        self.bundle = None

    def __str__(self):
        return f'{self.no}({self.pgleason})'

    @property
    def no(self):
        return int(self.info['Pro3 no'])

    @property
    def lgleasons(self):
        """Lesions Gleason scores."""
        keys = [f'GS_L{x}' for x in lrange()]
        vals = [self.info[x] for x in keys]
        return [value_default(GleasonScore.parse, x) for x in vals]

    @property
    def lgleason_max(self):
        """Maximum lesion Gleason score or None."""
        return max(list(filter(None, self.lgleasons)) or [None])

    @property
    def pgleason(self):
        """Prostate Gleason score or None."""
        keys = 'GS sum SB', 'GS sum TB'
        vals = [self.info[x] for x in keys]
        vals = [value_default(GleasonScore.parse, x) for x in vals]
        return max(list(filter(None, vals)) or [None])

    @property
    def likerts(self):
        """Likert grades."""
        keys = [f'Likert_L{i}' for i in lrange()]
        vals = [self.info[x] for x in keys]
        return [value_default(int, x) for x in vals]

    @property
    def likert_max(self):
        """Maximum Likert grade or None."""
        return max(list(filter(None, self.likerts)) or [None])


def read_csv_dicts(path, **kwargs):
    """Read columns from CSV as a dict of lists."""
    with open(path) as f:
        yield from csv.DictReader(f, **kwargs)


def read_patient_info(path='patients.tsv'):
    """Read patient info file."""
    it = (PatientInfo(x) for x in read_csv_dicts(path, delimiter='\t'))
    return {x.no: x for x in it}


def value_default(func, value, default=None):
    """Return `func(value)`, or `default` on `exception`."""
    try:
        return func(value)
    except (ValueError, TypeError):
        return default


def lrange():
    """Lesion indices."""
    return range(1, 5)


def nafilter(iterable, default=np.nan):
    """Filter out NAs; add `default` in case resulting iterable is empty."""
    it = filter(pd.notna, iterable)
    try:
        return chain([next(it)], it)
    except StopIteration:
        return iter([default])


def namax(iterable):
    """Return maximum value ignoring NAs, or NaN if given only NAs."""
    # return max(list(filter(pd.notna, values)) or [default])
    return max(nafilter(iterable))


def all_exist(seq):
    """Are all values in `seq1` not-NA and true?"""
    return all(bool(x) and pd.notna(x) for x in seq)


def read_patients_info_df(dataset='PRO3'):
    """Read and tidy patient info."""
    path = PATIENT_INFO_PATH[dataset]
    pats = pd.read_csv(path, sep='\t')
    pats = pats.drop(columns=['Data folder number', 'RALP no', 'Unnamed: 1',
                              'Name', 'DOB'])
    assert np.issubdtype(pats['Pro3 no'].dtype, np.int), pats['Pro3 no'].dtype
    pats = pats.set_index('Pro3 no')
    return pats


def convert_patient_info_df(pats):
    """Convert data types."""
    pats = pd.DataFrame(pats)

    def convert_to_gleason(obj):
        return value_default(GleasonScore.parse, str(obj), default=np.nan)
    keys = ['GS sum SB', 'GS sum TB'] + [f'GS_L{x}' for x in lrange()]
    pats[keys] = pats[keys].applymap(convert_to_gleason)

    def convert_to_likert(obj):
        return value_default(int, obj, default=np.nan)
    keys = [f'Likert_L{x}' for x in lrange()]
    pats[keys] = pats[keys].applymap(convert_to_likert).astype('Int8')

    return pats


def augment_patient_info_df(pats):
    """Add new columns."""
    # Prostate Gleason score is the maximum of SB & TB, if any.
    pats = pd.DataFrame(pats)
    keys = ['GS sum SB', 'GS sum TB']
    pats['GS_P'] = pats[keys].apply(namax, axis=1)
    keys = [f'GS_L{x}' for x in lrange()]
    pats['GS_Lmax'] = pats[keys].apply(namax, axis=1)
    keys = [f'Likert_L{x}' for x in lrange()]
    pats['Likert_Lmax'] = pats[keys].apply(namax, axis=1)
    return pats


def gleason_malign(value, threshold=(3, 3)):
    return int(value._astuple()[:2] > threshold)


def likert_malign(value):
    """Is Likert grade malign?"""
    assert value in range(1, 6), value
    return value > 2


# Read patient info
#   - Clean it
#   - Augment it
#   - Convert to case-indexed dicts.
# Read data bundles
# Find ROIs
#   - For each image:
#       - Read slice
#       - For each detection method:
#           - Detect candidates
#           - Take best candidate
#       - Take best overall
# Evaluate performance
#   - Calculate AUCs
#   - Calculate correlations
# Plot ROIs
#   - For each image:
#       - For each detection method:
#           - Plot image
#           - Plot mask outlines
#           - Plot ROI outlines


def correlations(roi_avgs, labels):
    roi_avgs = list(roi_avgs.values())
    def corr(x):
        # scale = dwi.stats.scale_standard
        return dwi.stats.correlation(x, labels, method='spearman')
    return [corr([x[i] for x in roi_avgs]) for i in range(3)]


def aucs(roi_avgs, labels):
    roi_avgs = list(roi_avgs.values())
    def auc(x):
        return dwi.stats.roc_auc(labels, x, autoflip=True)
    return [auc([x[i] for x in roi_avgs]) for i in range(3)]


pats = (read_patients_info_df()
        .pipe(convert_patient_info_df)
        .pipe(augment_patient_info_df))

targets = {x: ImageTarget(x, '', 1) for x in pats.index}
bundles = {k: dwi.readnib.ImageBundle(MODE, v) for k, v in targets.items()}
pats['has_ADC'] = [bundles[x].exists() for x in pats.index]
pats['use'] = pats[['GS_P', 'has_ADC']].apply(all_exist, axis=1)

pats = pats[pats['use'] == True]
pats['label33'] = pats['GS_P'].apply(lambda x: gleason_malign(x, (3, 3)))
pats['label34'] = pats['GS_P'].apply(lambda x: gleason_malign(x, (3, 4)))
pats['label_likert'] = pats['Likert_Lmax'].apply(likert_malign)

bundles = {k: v for k, v in bundles.items() if k in pats.index}
# res = [dwi.detectlesion.detect_blob(x, OUTDIR) for x in bundles.values()]
# roi_avgs = [x['roi_avgs'] for x in res]
blobs = {k: dwi.detectlesion.find_blobs(v.image_slice(), v.voxel_shape[0])
         for k, v in bundles.items()}
kwargs = dict(
    avg=np.nanmedian,
    # avg=np.nanmean,
    # avg=lambda x: np.nanpercentile(x, 50),
    )
rois = {k: [dwi.detectlesion.select_best_blob(v, x, **kwargs)
            for x in blobs[k]['blobs_list']]
        for k, v in bundles.items()}
roi_avgs = {k: [dwi.detectlesion.get_blob_avg(v, x, **kwargs) for x in rois[k]]
            for k, v in bundles.items()}

label_lists = [pats.label33, pats.label34]
cors33, cors34 = (correlations(roi_avgs, x) for x in label_lists)
aucs33, aucs34 = (aucs(roi_avgs, x) for x in label_lists)
