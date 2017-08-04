#!/usr/bin/python3

"""pat
pinky
image-hB image-lB
prostate-hB prostate-lB
lesion-hB lesion-lB
roi-hB roi-lB
"""

# from dwi.types import Path
from pathlib import Path

BASE = Path('/mri')


def nums(case):
    def num(pat):
        return len(list(BASE.glob(pat.format(**d))))

    r = {}
    d = dict(c=case)
    r['his'] = num('hist/New_ALL_RALP/{c}_*.*')
    for scan in ['1a', '1b', '2a', '2b']:
        d['s'] = scan
        r['img-hB-'+scan] = num('images/DWI/{c}-{s}*.*')
        r['img-lB-'+scan] = num('images/DWI_lB/{c}-{s}*.*')
        r['pro-hB-'+scan] = num('masks/prostate/DWI/{c}-{s}.*')
        r['pro-lB-'+scan] = num('masks/prostate/DWI_lB/{c}-{s}.*')
        r['les-hB-'+scan] = num('masks/lesion/DWI/lesion1/{c}-{s}.*')
        r['les-lB-'+scan] = num('masks/lesion/DWI_lB/lesion1/{c}-{s}.*')
        r['roi-hB-'+scan] = num('masks/roi/DWI/{c}-{s}.*')
        r['roi-lB-'+scan] = num('masks/roi/DWI_lB/{c}-{s}.*')

    r['img-T2'] = num('images/T2/{c}-*.*')
    r['img-T2w'] = num('images/T2w/{c}-*.*')

    r['pro-T2'] = num('masks/prostate/T2/{c}-*.*')
    r['pro-T2w'] = num('masks/prostate/T2w/{c}-*.*')

    r['les-T2'] = num('masks/lesion/T2/lesion1/{c}-*.*')
    r['les-T2w'] = num('masks/lesion/T2/lesion1/{c}-*.*')

    return r


def main():
    s = (
        '{case:3}'
        ', {his}'

        ', {img-hB-1a}, {img-hB-1b}, {img-hB-2a}, {img-hB-2b}'
        ', {pro-hB-1a}, {pro-hB-1b}, {pro-hB-2a}, {pro-hB-2b}'
        ', {les-hB-1a}, {les-hB-1b}, {les-hB-2a}, {les-hB-2b}'
        ', {roi-hB-1a}, {roi-hB-1b}, {roi-hB-2a}, {roi-hB-2b}'

        ', {img-lB-1a}, {img-lB-1b}, {img-lB-2a}, {img-lB-2b}'
        ', {pro-lB-1a}, {pro-lB-1b}, {pro-lB-2a}, {pro-lB-2b}'
        ', {les-lB-1a}, {les-lB-1b}, {les-lB-2a}, {les-lB-2b}'
        ', {roi-lB-1a}, {roi-lB-1b}, {roi-lB-2a}, {roi-lB-2b}'

        ', {img-T2}'
        ', {pro-T2}'
        ', {les-T2}'

        ', {img-T2w}'
        ', {pro-T2w}'
        ', {les-T2w}'
        )
    print('# ' + s.translate(s.maketrans('', '', '{}')))
    for case in range(350):
        d = nums(case)
        # if list(filter(None, d.values())):
        if d['his']:
            print(s.format(case=case, **d))


if __name__ == '__main__':
    main()
