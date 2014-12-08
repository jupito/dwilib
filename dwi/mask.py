"""ROI masks."""

import re
import numpy as np

class Mask(object):
    """Mask for one slice in 3D image."""
    def __init__(self, slice, array):
        if slice < 1:
            raise Exception('Invalid slice')
        self.slice = slice # Slice number, one-based indexing
        self.array = array.astype(bool) # 2D mask of one slice.

    def __repr__(self):
        return repr((self.slice, self.array.shape))

    def __str__(self):
        return repr(self)

    def get_subwindow(self, coordinates, onebased=True):
        """Get a view of a specific subwindow."""
        if onebased:
            coordinates = [i-1 for i in coordinates] # One-based indexing.
        z0, z1, y0, y1, x0, x1 = coordinates
        slice = self.slice - z0
        array = self.array[y0:y1,x0:x1]
        return Mask(slice, array)

    def get_masked(self, array):
        """Get masked region as a flat array."""
        if array.ndim == self.array.ndim:
            return array[self.array]
        else:
            return array[self.slice-1, self.array]

def load_ascii(filename):
    """Read a ROI mask file."""
    slice = 1
    arrays = []
    with open(filename, 'r') as f:
        p = re.compile(r'(\w+)\s*:\s*(.*)')
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = p.match(line)
            if m:
                if m.group(1) == 'slice':
                    slice = int(m.group(2))
            elif line[0] == '0' or line[0] == '1':
                a = np.array(list(line), dtype=int)
                arrays.append(a)
    if arrays:
        return Mask(slice, np.array(arrays))
    else:
        raise Exception('No mask found in %s' % filename)
