#!/usr/bin/python3

"""pat
pinky
image-hB image-lB
prostate-hB prostate-lB
lesion-hB lesion-lB
roi-hB roi-lB
"""

from collections import defaultdict

from tabulate import tabulate

from dwi.types import Path

BASE = Path('/mri')
LIST = 'misc/list/all.txt'


def read_cases():
    return sorted(int(x) for x in (BASE / LIST).read_text().split())


def nums(case):
    def num(pat):
        return len(list(BASE.glob(pat.format(**d))))

    r = defaultdict(int)
    d = dict(c=case)  # Used by num().

    r['case'] = case
    r['hist'] = num('hist/ALL_renamed_RALP/{c}_*.*')
    for rep, scan in ((x, y) for x in '12' for y in 'ab'):
        d['r'] = rep
        d['s'] = scan
        s = '-' + rep + scan
        r['hB-img' + s] += bool(num('images/DWI/{c}-{r}{s}*.*'))
        r['hB-Mono' + s] += bool(num('images/DWI-Mono/{c}-{r}{s}'))
        r['hB-Kurt' + s] += bool(num('images/DWI-Kurt/{c}-{r}{s}'))
        r['hB-pro' + s] += bool(num('masks/prostate/DWI/{c}-{r}{s}.*'))
        r['hB-les' + s] += bool(num('masks/lesion/DWI/lesion1/{c}-{r}{s}.*'))
        r['hB-roi' + s] += bool(num('masks/roi/DWI/{c}-{r}{s}_*'))

    for m in ['T2', 'T2w']:
        d['m'] = m
        r[m + '-img'] = num('images/{m}/{c}-*.*')
        r[m + '-pro'] = num('masks/prostate/{m}/{c}-*.*')
        r[m + '-les'] = num('masks/lesion/{m}/lesion1/{c}-*.*')

    # Keep only if something was found.
    if any(dict(r, case=None).values()):
        return r
    return None


def main():
    # cases = read_cases()
    cases = list(range(400))
    dicts = filter(None, (nums(x) for x in cases))
    fmt = 'tsv'
    s = tabulate(dicts, headers='keys', tablefmt=fmt)
    s = s.replace('\t', ', ')
    print(s)


if __name__ == '__main__':
    main()
