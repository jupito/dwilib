#!/usr/bin/python3

"""Calculate AUC for predicting prostate in image, lesion in prostate.

Written for the grid data.
"""

import argparse

import numpy as np

import dwi.files
import dwi.stats
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('--thresholds', '-t', nargs=2, type=float,
                   default=(0.5, 0.1),
                   help='thresholds for labeling as prostate, lesion')
    p.add_argument('pmaps', nargs='+',
                   help='input pmaps')
    return p.parse_args()


def main():
    args = parse_args()
    tuples = [dwi.files.read_pmap(x) for x in args.pmaps]
    pmaps = [x[0] for x in tuples]
    attrs = [x[1] for x in tuples]
    params = attrs[0]['parameters']
    if args.verbose > 1:
        print('Parameters:', params)
    if (any(x['parameters'] != params for x in attrs) or
        any(x.shape[-1] != len(params) for x in pmaps)):
        raise ValueError('Nonuniform parameters')

    # Flatten and concatenate pmaps.
    pmap = np.concatenate([x.reshape((-1, len(params))) for x in pmaps])
    if args.verbose > 1:
        print('Means:', np.mean(pmap, axis=0))

    # Sample values are in the last column.
    index = -1
    param = params[index]
    x = pmap[:, index]

    # Labels.
    assert params[:2] == ['prostate', 'lesion']
    y_prostate = pmap[:, 0] >= args.thresholds[0]
    y_lesion = pmap[:, 1] >= args.thresholds[1]

    # Average volume percentages.
    ppt = np.mean(y_prostate)  # Prostate per total.
    lpp = np.mean(y_lesion[y_prostate])  # Lesion per prostate.

    # ROC AUC of prostate against image.
    x = dwi.stats.scale_standard(x)
    _, _, auc_p = dwi.stats.calculate_roc_auc(y_prostate, x, autoflip=True)
    # ROC AUC of lesion against prostate.
    _, _, auc_l = dwi.stats.calculate_roc_auc(y_lesion[y_prostate],
                                              x[y_prostate], autoflip=True)

    if args.verbose:
        print('# l-auc p-auc p/total l/p p-threshold l-threshold n param')
    d = dict(p=param, pt=args.thresholds[0], lt=args.thresholds[1], n=len(x),
             ppt=ppt, lpp=lpp, la=auc_l, pa=auc_p)
    s = '{la:.3f}  {pa:.3f}  {ppt:.3f}  {lpp:.3f}  {pt}  {lt}  {n}  {p}'
    print(s.format(**d))


if __name__ == '__main__':
    main()
