"""Image as an xarray."""

# http://xarray.pydata.org/en/stable/

import logging

# import numpy as np
import xarray as xr

from . import files


def create(image, attrs, path=None):
    logging.warning('%s, %s', image.shape, image.dtype)
    logging.warning(attrs)
    dims = ('slc', 'col', 'row', 'param')
    image = xr.DataArray(image, dims=dims, name=path.name, attrs=attrs)
    if path is not None:
        image.attrs['path'] = str(path)
    return image


def create_dataset(images):
    return xr.merge(images)


def read(path, **kwargs):
    return create(*files.read_pmap(path, **kwargs), path=path)
