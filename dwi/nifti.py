"""Support for NIFTI files."""

import numpy as np
import nibabel


def read(path):
    """Read a NIFTI file."""
    img = nibabel.load(str(path))
    attrs = dict(img.header)
    a = img.get_data().astype(np.float32)
    assert a.ndim == 3, a.shape
    a = a.T  # Set axes to (depth, height, width).
    a.shape += (1,)  # Add parameter axis.
    return attrs, a
