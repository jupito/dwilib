"""Support for HDF5 files."""

from __future__ import division, print_function
import collections

import numpy as np
import h5py

DEFAULT_DSETNAME = 'array'

def write_hdf5(filename, array, attrs, dsetname=DEFAULT_DSETNAME):
    """Write an array with attributes into a newly created, compressed HDF5
    file."""
    f = h5py.File(filename, 'w')
    dset = f.create_dataset(dsetname, data=array, compression='gzip',
            shuffle=True, fletcher32=True)
    for k, v in attrs.iteritems():
        dset.attrs[k] = v
    f.close()

def read_hdf5(filename, dsetname=DEFAULT_DSETNAME):
    """Read an array with attributes from an HDF5 file."""
    f = h5py.File(filename, 'r')
    dset = f[dsetname]
    array = np.array(dset)
    attrs = collections.OrderedDict(dset.attrs)
    f.close()
    return array, attrs
