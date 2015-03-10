import os
import re
import numpy as np

import util

# Handle signal intensity and parameter map files as ASCII.

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
            a = util.fabricate_subwindow(len(self.a))
        return tuple(map(int, a))

    def subwinsize(self):
        # XXX: Remove in favor of subwindow_shape()?
        a = self.subwindow()
        r = []
        for i in range(len(a)/2):
            r.append(a[i*2+1] - a[i*2])
        return tuple(r)

    def subwindow_shape(self):
        return util.subwindow_shape(self.subwindow())

    def bset(self):
        """Return the b-value set. Fabricate if not present."""
        a = re.findall(r'[\d.]+', self.d.get('bset', ''))
        if not a:
            a = range(len(self.a[0]))
        return tuple(map(float, a))

    def params(self):
        r = range(self.a.shape[1])
        r = map(str, r)
        a = re.findall(r'\S+', self.d.get('parameters', ''))
        print a
        for i, s in enumerate(a):
            r[i] = s
        return tuple(r)

    def number(self):
        m = re.search(r'\d+', self.d.get('number', ''))
        return int(m.group()) if m else 1

    def roislice(self):
        return self.d.get('ROIslice', '')

    def name(self):
        return self.d.get('name', '')

def read_ascii(f):
    d = {}
    rows = []
    pvar = re.compile(r'(\w+)\s*:\s*(.*)')
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            continue
        var = pvar.match(line)
        if var:
            d[var.group(1)] = var.group(2)
        else:
            try:
                nums = map(float, line.split())
            except ValueError as e:
                #print e
                #continue
                raise
            if nums:
                rows.append(nums)
    rows = np.array(rows, dtype=np.float)
    return d, rows

def read_ascii_file(filename):
    with open(filename, 'rU') as f:
        return read_ascii(f)

def write_ascii_file(filename, pmap, params):
    """Write parametric map in ASCII format."""
    with open(filename, 'w') as f:
        f.write('parameters: %s\n' % ' '.join(map(str, params)))
        for values in pmap:
            f.write(' '.join(map(repr, values)) + '\n')
