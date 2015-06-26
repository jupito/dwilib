"""Support for HDF5 files."""

import numpy as np
import h5py

def write_hdf5(filename, array, attrs, dsetname='array'):
    f = h5py.File(filename, 'w')
    dset = f.create_dataset(dsetname, data=array, compression='gzip',
            shuffle=True, fletcher32=True)
    dset.attrs.update(attrs)
    f.close()

def read_hdf5(filename, dsetname='array'):
    f = h5py.File(filename, 'r')
    dset = f[dsetname]
    array = np.array(dset)
    print dset.attrs
    attrs = dset.attrs
    return array, attrs
