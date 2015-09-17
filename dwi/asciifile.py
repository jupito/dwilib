"""Handle signal intensity and parameter map files as ASCII.

Use read_ascii_file() and write_ascii_file() to read and write.
"""

from __future__ import absolute_import, division, print_function
import os
import re

import numpy as np

import dwi.files
import dwi.util


class AsciiFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.d, self.a = read_ascii_file(self.filename)
        self.number = int(self.d.get('number', 0))

    def __repr__(self):
        return self.filename

    def __str__(self):
        return '{}\n{}\n{}'.format(self.filename, self.d, self.a.shape)

    def subwindow(self):
        a = re.findall(r'\d+', self.d.get('subwindow', ''))
        if not a:
            a = dwi.util.fabricate_subwindow(len(self.a))
        return tuple(int(x) for x in a)

    def subwinsize(self):
        # TODO: Remove in favor of subwindow_shape()?
        a = self.subwindow()
        r = []
        for i in range(len(a)//2):
            r.append(a[i*2+1] - a[i*2])
        return tuple(r)

    def subwindow_shape(self):
        return dwi.util.subwindow_shape(self.subwindow())

    def bset(self):
        """Return the b-value set. Fabricate if not present."""
        a = re.findall(r'[\d.]+', self.d.get('bset', ''))
        if not a:
            a = range(len(self.a[0]))
        return tuple(float(x) for x in a)

    def params(self):
        r = range(self.a.shape[1])
        r = [str(x) for x in r]
        a = re.findall(r'\S+', self.d.get('parameters', ''))
        for i, s in enumerate(a):
            r[i] = s
        return tuple(r)

    def roislice(self):
        return self.d.get('ROIslice', '')

    def name(self):
        return self.d.get('name', '')


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


def write_ascii_file(filename, pmap, params):
    """Write parametric map in ASCII format."""
    with open(filename, 'w') as f:
        f.write('parameters: %s\n' % ' '.join(str(x) for x in params))
        for values in pmap:
            if len(values) != len(params):
                raise Exception('Number of values and parameters mismatch')
            f.write(' '.join(repr(x) for x in values) + '\n')
