"""Support for HDF5 files."""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np
import h5py

import dwi.util

DEFAULT_DSETNAME = 'default'
DEFAULT_DSETPARAMS = dict(
    compression='gzip',  # Smaller, compatible.
    # compression='lzf',  # Faster.
    shuffle=True,  # Rearrange bytes for better compression.
    fletcher32=True,  # Flether32 checksum.
    track_times=False,  # Dataset creation timestamps.
    )


class Dataset(h5py.Dataset):
    """Add some missing ndarray API to dataset."""
    # TODO: Not needed anymore?
    @property
    def ndim(self):
        return len(self.shape)


def convert_value_write(v):
    """HDF5 doesn't understand None objects, so replace any with nan values."""
    def convert_item(x):
        """Convert sequence item."""
        if x is None:
            return np.nan
        if dwi.util.isstring(x):
            return x.encode()
        return x
    if dwi.util.iterable(v) and not dwi.util.isstring(v):
        # if any(x is None for x in v):
        #     v = type(v)([(np.nan if x is None else x) for x in v])
        v = type(v)([convert_item(x) for x in v])
    return v


def write_hdf5(filename, array, attrs, fillvalue=None,
               dsetname=DEFAULT_DSETNAME):
    """Write an array with attributes into a newly created, compressed HDF5
    file.
    """
    f = h5py.File(filename, 'w')
    dset = f.create_dataset(dsetname, data=array, fillvalue=fillvalue,
                            **DEFAULT_DSETPARAMS)
    for k, v in attrs.items():
        dset.attrs[k] = convert_value_write(v)
    f.close()


def convert_value_read(value):
    """Convert attribute value from bytes to string."""
    if isinstance(value, bytes):
        return value.decode()
    elif not np.isscalar(value) and np.issubsctype(value, np.bytes_):
        return value.astype(np.str_)
    return value


def convert_attrs_read(attrs):
    """Convert attribute values from bytes to strings."""
    return ((k, convert_value_read(v)) for k, v in attrs.items())


def read_hdf5(filename, ondisk=False, dsetname=DEFAULT_DSETNAME):
    """Read an array with attributes from an HDF5 file.

    With parameter "ondisk" True it will not be read into memory."""
    try:
        f = h5py.File(filename, 'r')
    except IOError as e:
        if e.filename is None:
            e.filename = filename
        raise
    if dsetname not in f:
        # No dataset of given name, try the one there is.
        try:
            dsetname, = f.keys()
        except ValueError:
            raise ValueError('Ambiguous content: {}'.format(filename))
    dset = f[dsetname]
    if ondisk:
        array = Dataset(dset.id)
    else:
        array = np.array(dset)
    attrs = OrderedDict(dset.attrs)
    attrs.update(convert_attrs_read(attrs))
    if not ondisk:
        f.close()
    return array, attrs


def create_hdf5(filename, shape, dtype, fillvalue=None,
                dsetname=DEFAULT_DSETNAME):
    """Create a HDF5 file and return the dataset for manipulation.

    Attributes and the file object can be accessed by dset.attrs and dset.file.
    """
    f = h5py.File(filename, 'w')
    dset = f.create_dataset(dsetname, shape, dtype=dtype, fillvalue=fillvalue,
                            **DEFAULT_DSETPARAMS)
    return Dataset(dset.id)
