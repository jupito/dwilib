#!/usr/bin/python3

"""Produce inventory of all data."""

# import logging
from collections import defaultdict

from tabulate import tabulate

from dwi import files
from dwi.types import Path

BASE = Path('/mri')
LIST = 'misc/list/all.txt'


def read_cases():
    # XXX: Not used anymore.
    return sorted(int(x) for x in (BASE / LIST).read_text().split())


def nums_str(case):
    def num(pat):
        n = len(list(BASE.glob(pat.format(**d))))
        d['exist'] += n
        return n

    def ss(n):
        """Return number as string if nonzero, else slash."""
        return str(n or '-')

    r = defaultdict(str)
    d = dict(c=case, exist=0)  # Used by num().

    r['case'] = case
    r['hist'] = ss(num('hist/ALL_renamed_RALP/{c}_*.*'))
    for rep, scan in ((x, y) for x in '12' for y in 'ab'):
        d['r'] = rep
        d['s'] = scan
        # t = rep + scan  # "True"
        # f = '-' * len(t)  # "False"
        # r['hB-img'] += t if num('images/DWI/{c}-{r}{s}*.*') else f
        # r['hB-Mono'] += t if num('images/DWI-Mono/{c}-{r}{s}') else f
        # r['hB-Kurt'] += t if num('images/DWI-Kurt/{c}-{r}{s}') else f
        r['hB-img'] += ss(num('images/DWI/{c}-{r}{s}*.*'))
        r['hB-Mono'] += ss(num('images/DWI-Mono/{c}-{r}{s}'))
        r['hB-Kurt'] += ss(num('images/DWI-Kurt/{c}-{r}{s}'))

        # r['hB-pro'] += t if num('masks/prostate/DWI_hB*/{c}-{r}{s}_*.*') else f
        r['hB-pro'] += ss(num('masks/prostate/DWI_hB*/{c}-{r}{s}*.*'))
        r['hB-les'] += ss(num('masks/lesion/DWI_hB/lesion?/{c}-{r}{s}.*'))
        r['hB-5x5'] += ss(num('masks/roi/DWI_hB/{c}-{r}{s}_*'))

        # r['lB-img'] += t if num('images/DWI_lB/{c}-{r}{s}*.*') else f
        # r['lB-pro'] += t if num('masks/prostate/DWI_lB/{c}-{r}{s}.*') else f
        r['lB-img'] += ss(num('images/DWI_lB/{c}-{r}{s}*.*'))
        r['lB-pro'] += ss(num('masks/prostate/DWI_lB/{c}-{r}{s}.*'))
        r['lB-5x5'] += ss(num('masks/roi/DWI_lB/{c}-{r}{s}_*'))

    for m in ['T2', 'T2w']:
        d['m'] = m
        r[m + '-img'] = ss(num('images/{m}/{c}-*.*'))
        r[m + '-pro'] = ss(num('masks/prostate/{m}/{c}-*.*'))
        r[m + '-les'] = ss(num('masks/lesion/{m}/lesion?/{c}-*.*'))

    if d['exist']:
        return r
    return None


def add_score(dicts):
    """Add Gleason score."""
    path = BASE / 'work/patients/patients_DWI_67.txt'
    patients = files.read_patients_file(path)

    def max_score(patient):
        return max(x.score for x in patient.lesions)

    def all_scores(patient):
        return '/'.join(str(x.score) for x in patient.lesions)

    for d in dicts:
        s = '-'
        for p in patients:
            if p.num == d['case']:
                # s = str(max_score(p))
                s = all_scores(p)
                break
        d['GS'] = s


def main():
    # cases = read_cases()
    cases = list(range(400))
    dicts = list(filter(None, (nums_str(x) for x in cases)))
    add_score(dicts)

    fmt = 'tsv'
    s = tabulate(dicts, headers='keys', tablefmt=fmt)
    s = s.replace('\t', ', ')
    print(s)
    assert len(s.split('\n')) == len(dicts) + 1


if __name__ == '__main__':
    main()
