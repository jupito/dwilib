"""Image as an xarray."""

# http://xarray.pydata.org/en/stable/

import logging

# import numpy as np
import xarray as xr

from . import files


# class XImage(xr.DataArray):
#     @classmethod
#     def read(cls, path):
#         pmap, attrs = files.read_pmap(path, **kwargs)
#         return clr(


def create(image, attrs, path):
    logging.info('%s, %s', image.shape, image.dtype)
    logging.info(attrs)
    # dims = 'slc', 'col', 'row', 'param'
    dims = 'z', 'y', 'x', 'param'
    image = xr.DataArray(image, dims=dims, name=path.name, attrs=attrs)
    image.attrs['path'] = str(path)
    return image


def create_dataset(images):
    return xr.merge(images)


def read(path, **kwargs):
    return create(*files.read_pmap(path, **kwargs), path=path)


def read_mask(path, **kwargs):
    return create([files.read_mask(path, **kwargs), {}], path=path)
