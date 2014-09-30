#!/usr/bin/env python

# Print information columns of number values.

import sys
import numpy as np

from dwi import asciifile
from dwi import util

for filename in sys.argv[1:]:
    af = asciifile.AsciiFile(filename)
    print filename
    if af.d.has_key('description'):
        print af.d['description']
    params = af.params()
    for i, a in enumerate(af.a.T):
        d = dict(i=i, param=params[i],
                median=np.median(a), mean=a.mean(), var=a.var(),
                min=a.min(), max=a.max(), sum=sum(a))
        print '{i}\t{param}'\
                '\t{median:g}\t{mean:g}\t{var:g}'\
                '\t{min:g}\t{max:g}\t{sum:g}'.format(**d)
        print util.fivenum(a)
