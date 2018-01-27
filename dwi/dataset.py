"""Dataset handling."""

# from functools import lru_cache
import logging

import numpy as np

from . import files, image, paths
from .types import ImageMode, Lesion, Path, TextureSpec
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


class DummyDataset(Dataset):
    def __init__(self, mode, samplelist, cases=None):
        self.mode = ImageMode(mode)
        self.samplelist = samplelist
        self.cases = cases

    @property
    def samplelist_path(self):
        raise NotImplementedError()

    def each_patient(self):
        raise NotImplementedError()

    def _cases(self):
        return range(300)

    def _scans(self):
        return (x + y for x in '12' for y in 'ab')

    def _lesions(self):
        return (Lesion(x, '0+0', 'NA') for x in range(3))

    def each_image_id(self):
        for case in self.cases or self._cases():
            for scan in self._scans():
                yield case, scan, list(self._lesions())


# TODO: A new attempt!
class ImageData(object):
    """Generate and easily access all images, masks, gleason scores, etc.

    Combinations of values are indicated with lists of keys. Valid keys are
    mode, case, scan, lesion, masktype.
    """
    def __init__(self, modes, cases, scans=None, maxlesions=3, masktypes=None,
                 base='.'):
        if scans is None:
            scans = [x + y for x in '12' for y in 'ab']
        if masktypes is None:
            masktypes = ['prostate', 'lesion']
        self.choices = dict(
            mode=[ImageMode(x) for x in modes],
            case=list(cases),
            scan=list(scans),
            lesion=list(range(maxlesions)),
            masktype=list(masktypes),
        )
        self.base = Path(base)

    @property
    def valid_keys(self):
        return list(self.choices.keys())

    def combinations(self, keys):
        """Generate all value combinations for `keys` (in given order)."""
        yield from self._combinations(keys, ImageDataTarget(base=self.base))

    def _combinations(self, keys, output):
        """Generate all value combinations for `keys` (in given order).

        Dictionary `output` is copied and passed recursively to keep state.
        """
        logging.debug([keys, output])
        if keys:
            key, rest = keys[0], keys[1:]
            values = self.choices[key]
            assert values, key
            for value in values:
                d = ImageDataTarget(output)
                d[key] = value
                yield from self._combinations(rest, d)
        else:
            yield output


class ImageDataTarget(dict):
    """Target definition yielded by ImageData."""
    @property
    def image_path(self):
        p = paths.Paths(self['mode'], base=self['base'])
        return p.pmap(self['case'], self['scan'])

    @property
    def mask_path(self):
        p = paths.Paths(self['mode'], base=self['base'])
        return p.mask(self['masktype'], self['case'], self['scan'],
                      lesion=self.get('lesion'))

    @property
    def histology_path(self):
        p = paths.Paths('', base=self['base'])
        return p.histology(self['case'])


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
