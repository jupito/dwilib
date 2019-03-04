"""Reading by nibabel."""

# http://nipy.org/nibabel/gettingstarted.html

import logging
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

import dwi.mask
import dwi.util
# from dwi.types2 import ImageMode, ImageTarget

DEFAULT_ROOT = '~/tmp/data/Data_Organized_29012019'
DEFAULT_SUFFIX = '.nii'
DEFAULT_COLLECTION = 'IMPROD'
DEFAULT_PSTEM = 'PM'
DEFAULT_LSTEM = 'LS'
# DEFAULT_VOXEL_SHAPES = dict(DWI=(2., 2., 3.), T2w=(0.625, 0.625, 3.))
DEFAULT_VOXEL_SHAPES = dict(DWI=(3., 2., 2.), T2w=(3., 0.625, 0.625))

log = logging.getLogger(__name__)


class ImageBundle:
    """A bundle of (image, pmask, lmask) for (mode, case) in NIFTI format."""
    def __init__(self, mode, target, root=DEFAULT_ROOT,
                 suffix=DEFAULT_SUFFIX, collection=DEFAULT_COLLECTION):
        self.mode = mode
        self.target = target
        self._root = Path(root).expanduser()
        self._suffix = suffix
        self._collection = collection

    def _modalitydir(self):
        return (self._root /
                f'{self._collection}_{self.mode.modality}_human_drawn')

    def _imagedir(self):
        return (self._modalitydir() /
                f'{self.target.case}_L{self.target.lesion}')

    def _path(self, stem):
        return (self._imagedir() / stem).with_suffix(self._suffix)

    def _load(self, stem):
        return nib.load(str(self._path(stem)))

    @staticmethod
    @lru_cache(None)
    def _ornt():
        """Get orientation change for `as_reoriented()`."""
        # TODO: Get orientation info from NIFTI headers.
        ornt = np.zeros((3, 2), dtype=np.int8)
        # ornt[:, 0] = range(3)  # No transpose.
        ornt[:, 0] = 2, 1, 0  # Transpose to (slice, vertical, horizontal).
        ornt[:, 1] = -1, 1, 1  # Flip axis 0 (up-down).
        return ornt

    def _print_debug_info(self):
        lst = [self.image, self.pmask, self.lmask]
        print(self.target, self.voxel_shape(), self.image.shape,
              [x.get_data_dtype() for x in lst],
              [x.affine.shape for x in lst],
              [x.header.get_slope_inter() for x in lst],
              sep='\n\t')
        # print(*self.image.header.items(), sep='\n\t')
        # fdata = self.image.get_fdata()

    def exists(self, stem=DEFAULT_LSTEM):
        return self._path(stem).exists()  # For simplicity check only lesion.

    @property
    def shape(self):
        return self.image.shape

    @property
    def dtype(self):
        return self.image.get_data_dtype()

    def voxel_shape(self):
        """Get voxel shape in millimetres."""
        shape = self.image.header.get_zooms()[:3]
        default = DEFAULT_VOXEL_SHAPES[self.mode.modality[:3]]
        assert ([round(x, 1) for x in shape] ==
                [round(x, 1) for x in default]), shape
        return shape

    def voxel_size(self):
        """Get voxel width, which is assumed to be equal to height."""
        shape = self.voxel_shape()
        assert shape[1] == shape[2], shape
        return shape[1]

    @property
    def image(self):
        return self._load(self.mode.param).as_reoriented(self._ornt())

    @property
    def pmask(self):
        return self._load(DEFAULT_PSTEM).as_reoriented(self._ornt())

    @property
    def lmask(self):
        return self._load(DEFAULT_LSTEM).as_reoriented(self._ornt())

    @property
    def image_data(self):
        return self.image.get_fdata()

    @property
    def pmask_data(self):
        return self.pmask.get_fdata().clip(0, 1)

    @property
    def lmask_data(self):
        return self.lmask.get_fdata().clip(0, 1)

    @lru_cache(None)
    def p_mbb(self, pad=5):
        return dwi.util.bbox(self.pmask_data, pad=(np.inf, pad, pad))

    @lru_cache(None)
    def p_max_slice_index(self):
        """Get the slice index with maximum prostate area."""
        mask = dwi.mask.Mask3D(self.pmask_data[self.p_mbb()])
        return dwi.util.middle(mask.max_slices())

    def image_slice(self):
        """Get the image slice with maximum prostate area."""
        return self.image_data[self.p_mbb()][self.p_max_slice_index()]

    def pmask_slice(self):
        """Get the prostate mask slice with maximum prostate area."""
        return self.pmask_data[self.p_mbb()][self.p_max_slice_index()]

    def lmask_slice(self):
        """Get the lesion mask slice with maximum prostate area."""
        return self.lmask_data[self.p_mbb()][self.p_max_slice_index()]

    def pmasked_image_slice(self):
        masked = self.image_slice().copy()
        masked[~self.pmask_slice().astype(np.bool)] = np.nan
        return masked

#     def pmasked_image_slice_(self):
#         masked = self.pmasked_image_slice()
#         masked[np.isnan(masked)] = np.random.random_sample(masked.shape).clip(np.nanmin(masked), np.nanmax(masked))[np.isnan(masked)]
#         assert not np.any(np.isnan(masked))
#         return masked


# def load_images(case, modes):
#     # targets = [ImageTarget(case, '', x) for x in [10]]  # Combined lesions.
#     targets = [ImageTarget(case, '', x) for x in [1]]
#     bundles = [ImageBundle(m, t) for m in modes for t in targets]
#     return bundles


# def get_slices(bundle):
#     """From image & masks, get maximum lesion slice, prostate bounding box."""
#     # pad = [x // 10 for x in bundle.image.shape]
#     pad = 5
#     mbb = dwi.util.bbox(bundle.pmask_data, pad=pad)
#     # Use maximum slice in prostate mask.
#     i = dwi.mask.Mask3D(bundle.pmask_data[mbb]).max_slices()[0]
#     return i, [x[mbb][i] for x in [bundle.image_data, bundle.pmask_data,
#                                    bundle.lmask_data]]
