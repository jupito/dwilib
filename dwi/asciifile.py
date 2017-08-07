"""Handle signal intensity and parameter map files as ASCII.

Use read_ascii_file() and write_ascii_file() to read and write.
"""

# NOTE: Obsolete, don't use for new code.

import os
import re

import numpy as np

import dwi.files
import dwi.util


class AsciiFile(object):
    def __init__(self, filename):
        filename = str(filename)
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.d, self.a = read_ascii_file(self.filename)
        self.number = int(self.d.get('number', 0))
        self.roislice = self.d.get('ROIslice', '')
        self.name = self.d.get('name', '')

    def __repr__(self):
        return self.filename

    def __str__(self):
        return '{}\n{}\n{}'.format(self.filename, self.d, self.a.shape)

    # def subwindow(self):
    #     a = re.findall(r'\d+', self.d.get('subwindow', ''))
    #     if not a:
    #         a = dwi.util.fabricate_subwindow(len(self.a))
    #     return tuple(int(x) for x in a)

    # def bset(self):
    #     """Return the b-value set. Fabricate if not present."""
    #     a = re.findall(r'[\d.]+', self.d.get('bset', ''))
    #     if not a:
    #         a = range(len(self.a[0]))
    #     return tuple(float(x) for x in a)

    def params(self):
        r = range(self.a.shape[1])
        r = [str(x) for x in r]
        a = re.findall(r'\S+', self.d.get('parameters', ''))
        for i, s in enumerate(a):
            r[i] = s
        return tuple(r)


def read_ascii_file(filename):
    d = {}
    rows = []
    pvar = re.compile(r'(\w+)\s*:\s*(.*)')
    for line in dwi.files.valid_lines(filename):
        var = pvar.match(line)
        if var:
            d[var.group(1)] = var.group(2)
        else:
            nums = [float(x) for x in line.split()]
            if nums:
                rows.append(nums)
    rows = np.array(rows, dtype=np.float64)
    return d, rows


def write_ascii_file(filename, pmap, params, attrs=None):
    """Write parametric map in ASCII format."""
    if params is not None and attrs is None:
        attrs = dict(parameters=params)
    with open(str(filename), 'w') as f:
        for k, v in sorted(attrs.items()):
            if isinstance(v, (list, np.ndarray)):
                v = ' '.join(str(x) for x in v)
            f.write('{k}: {v}\n'.format(k=k, v=v))
        for values in pmap:
            if len(values) != len(attrs['parameters']):
                raise Exception('Number of values and parameters mismatch')
            f.write(' '.join(repr(x) for x in values) + '\n')
