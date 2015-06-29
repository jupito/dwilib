"""Support for HDF5 files."""

import collections

import numpy as np
import h5py

DEFAULT_DSETNAME = 'array'

def write_hdf5(filename, array, attrs, dsetname=DEFAULT_DSETNAME):
    f = h5py.File(filename, 'w')
    dset = f.create_dataset(dsetname, data=array, compression='gzip',
            shuffle=True, fletcher32=True)
    dset.attrs.update(attrs)
    f.close()

def read_hdf5(filename, dsetname=DEFAULT_DSETNAME):
    f = h5py.File(filename, 'r')
    dset = f[dsetname]
    array = np.array(dset)
    attrs = collections.OrderedDict(dset.attrs)
    f.close()
    return array, attrs
