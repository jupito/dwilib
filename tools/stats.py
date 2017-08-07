#!/usr/bin/python3

"""List lesion statistics."""

from collections import OrderedDict
from itertools import product

import dwi.dataset
import dwi.image
from dwi.types import ImageMode, Path
import numpy as np
from tabulate import tabulate

BASE = Path('/mri')
LIST = 'misc/list/all.txt'


def read_cases():
    return sorted(int(x) for x in (BASE / LIST).read_text().split())


def read_lmasks(paths, case, scan):
    for les in [1, 2, 3]:
        try:
            yield dwi.image.Image.read_mask(paths.mask('lesion', case, scan,
                                                       les))
        except OSError:
            yield None


def get_image_stats(paths, case, scan):
    try:
        pmap = dwi.image.Image.read(paths.pmap(case, scan))
    except FileNotFoundError:
        return None
    # pmask = dwi.image.Image.read_mask(paths.mask('prostate', case, scan))
    lmasks = list(read_lmasks(paths, case, scan))
    # print(case, scan, len(lmasks))
    d = OrderedDict()
    d['case'] = case
    d['scan'] = scan
    for i, lmask in enumerate(lmasks):
        if lmask is not None:
            def key(s):
                return '{}-{}'.format(i+1, s)
            masked = pmap[lmask]
            slices = [p[m] for p, m in zip(pmap, lmask)]
            slices = [x for x in slices if x.size]
            d[key('size')] = masked.size
            d[key('slices')] = len(slices)
            d[key('mean')] = np.mean(masked)
            d[key('median')] = np.median(masked)
            d[key('medmed')] = np.median([np.median(x) for x in slices])
    return d


def main():
    mode = ImageMode('DWI-Mono-ADCm')
    paths = dwi.paths.Paths(mode)
    cases = read_cases()
    cases = cases[:5]
    it = product(cases, [1, 2], 'ab')
    it = ((c, '{}{}'.format(r, s)) for c, r, s in it)
    # for case, _scan1, _scan2 in it:
    #     scan = '{}{}'.format(_scan1, _scan2)
    #     get_image_stats(paths, case, scan)
    it = (get_image_stats(paths, c, s) for c, s in it)
    it = filter(None, it)
    d = dict(headers='keys', tablefmt='tsv', floatfmt='1.5f', missingval='-')
    s = tabulate(it, **d)
    s = s.replace('\t', ', ')
    print(s)


if __name__ == '__main__':
    main()
