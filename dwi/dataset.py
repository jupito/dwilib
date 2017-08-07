"""Dataset handling."""

# from functools import lru_cache
import logging

import numpy as np

from . import files, image, paths
from .types import ImageMode, Lesion, TextureSpec
from .util import unify_masks


class Dataset(object):
    """Dataset class for easier access to patients with images."""
    def __init__(self, mode, samplelist, cases=None):
        self.mode = ImageMode(mode)
        self.samplelist = samplelist
        self.cases = cases

    @property
    def samplelist_path(self):
        return paths.samplelist_path(self.mode, self.samplelist)

    def each_patient(self):
        """Generate patients."""
        patients = files.read_patients_file(self.samplelist_path)
        for p in patients:
            if self.cases is None or p.num in self.cases:
                yield p

    def each_image_id(self):
        for p in self.each_patient():
            for s in p.scans:
                yield p.num, s, p.lesions

    def each_lesion(self):
        """Generate lesions."""
        for case, scan, lesions in self.each_image_id():
            for lesion in lesions:
                yield case, scan, lesion


# @lru_cache(maxsize=16)
def read_prostate_mask(mode, case, scan):
    """Read prostate mask."""
    path = paths.mask_path(mode, 'prostate', case, scan)
    return image.Image.read_mask(path)


def read_lesion_mask(mode, case, scan, lesion):
    """Read lesion mask."""
    if isinstance(lesion, Lesion):
        lesion = lesion.index + 1
    path = paths.mask_path(mode, 'lesion', case, scan, lesion=lesion)
    return image.Image.read_mask(path)


# @lru_cache(maxsize=16)
def read_lesion_masks(mode, case, scan, lesions, only_largest=False):
    """Read and combine multiple lesion masks (for same scan)."""
    masks = (read_lesion_mask(mode, case, scan, x) for x in lesions)
    if only_largest:
        # Use only the biggest lesion.
        d = {np.count_nonzero(x): x for x in masks}
        masks = [d[max(d.keys())]]
        logging.warning([mode, case, scan, lesions, d.keys(),
                         np.count_nonzero(masks[0])])
    return unify_masks(masks)


def iterlesions(patients):
    """Generate all case, scan, lesion combinations."""
    if isinstance(patients, str):
        patients = files.read_patients_file(patients)
    for p in patients:
        for s in p.scans:
            for l in p.lesions:
                yield p, s, l


def read_tmap(mode, case, scan, tspec=None, masktype='prostate', **kwargs):
    """Read a texture map."""
    if tspec is None:
        tspec = TextureSpec('raw', 1, None)
    path = paths.texture_path(mode, case, scan, None, masktype, 'all', 0,
                              tspec, voxel='all')
    return image.Image.read(path, **kwargs)
