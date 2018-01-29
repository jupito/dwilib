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
    r['case'] = case
    d = dict(c=case)  # Used by num().

    r['hist'] = num('hist/ALL_renamed_RALP/{c}_*.*')
    # for scan in ['1a', '1b', '2a', '2b']:
    #     d['s'] = scan
    #     r['img-hB-'+scan] = num('images/DWI/{c}-{s}*.*')
    #     r['img-lB-'+scan] = num('images/DWI_lB/{c}-{s}*.*')
    #     r['pro-hB-'+scan] = num('masks/prostate/DWI/{c}-{s}.*')
    #     r['pro-lB-'+scan] = num('masks/prostate/DWI_lB/{c}-{s}.*')
    #     r['les-hB-'+scan] = num('masks/lesion/DWI/lesion1/{c}-{s}.*')
    #     r['les-lB-'+scan] = num('masks/lesion/DWI_lB/lesion1/{c}-{s}.*')
    #     r['roi-hB-'+scan] = num('masks/roi/DWI/{c}-{s}.*')
    #     r['roi-lB-'+scan] = num('masks/roi/DWI_lB/{c}-{s}.*')
    for rep in [1, 2]:
        d['r'] = rep
        r['hB-img'] += (num('images/DWI/{c}-{r}[ab]*.*') > 1)
        r['hB-Mono'] += (num('images/DWI-Mono/{c}-{r}[ab]') > 1)
        r['hB-Kurt'] += (num('images/DWI-Kurt/{c}-{r}[ab]') > 1)
        r['hB-pro'] += (num('masks/prostate/DWI/{c}-{r}[ab].*') > 1)
        r['hB-les'] += (num('masks/lesion/DWI/lesion1/{c}-{r}[ab].*') > 1)
        r['hB-roi'] += (num('masks/roi/DWI/{c}-{r}[ab]_*') > 1)

    for m in ['T2', 'T2w']:
        d['m'] = m
        r[m + '-img'] = num('images/{m}/{c}-*.*')
        r[m + '-pro'] = num('masks/prostate/{m}/{c}-*.*')
        r[m + '-les'] = num('masks/lesion/{m}/lesion1/{c}-*.*')
    d.pop('m')

    if any(dict(r, case=None).values()):
        return r
    return None


def main():
    # cases = read_cases()
    cases = list(range(400))
    # print('{}: {}'.format(len(cases), ' '.join(str(x) for x in cases)))

    # s = (
    #     '{case:3}'
    #     ', {hist}'

    #     # ', {img-hB-1a}, {img-hB-1b}, {img-hB-2a}, {img-hB-2b}'
    #     # ', {pro-hB-1a}, {pro-hB-1b}, {pro-hB-2a}, {pro-hB-2b}'
    #     # ', {les-hB-1a}, {les-hB-1b}, {les-hB-2a}, {les-hB-2b}'
    #     # ', {roi-hB-1a}, {roi-hB-1b}, {roi-hB-2a}, {roi-hB-2b}'

    #     # ', {img-lB-1a}, {img-lB-1b}, {img-lB-2a}, {img-lB-2b}'
    #     # ', {pro-lB-1a}, {pro-lB-1b}, {pro-lB-2a}, {pro-lB-2b}'
    #     # ', {les-lB-1a}, {les-lB-1b}, {les-lB-2a}, {les-lB-2b}'
    #     # ', {roi-lB-1a}, {roi-lB-1b}, {roi-lB-2a}, {roi-lB-2b}'

    #     ', {img-hB:d}'
    #     ', {pro-hB:d}'
    #     ', {les-hB:d}'
    #     ', {roi-hB:d}'

    #     # ', {img-T2}'
    #     # ', {pro-T2}'
    #     # ', {les-T2}'

    #     # ', {img-T2w}'
    #     # ', {pro-T2w}'
    #     # ', {les-T2w}'
    # )
    # print('# ' + s.translate(s.maketrans('', '', '{}')))

    dicts = filter(None, (nums(x) for x in cases))

    # for d in dicts:
    #     # if d['hist'] and len(list(filter(None, d.values()))) > 2:
    #     # if len(list(filter(None, d.values()))) >= 5:
    #     # if all(bool(d[x]) for x in 'img-hB pro-hB roi-hB'.split()):
    #     #     print(s.format(**d))
    #     print(s.format(**d))
    # fmt = 'plain'
    fmt = 'tsv'
    s = tabulate(dicts, headers='keys', tablefmt=fmt)
    s = s.replace('\t', ', ')
    print(s)


if __name__ == '__main__':
    main()
