#!/usr/bin/python3

"""Plot boxplots of images. Possible nans and infinities are ignored."""

import argparse
from collections import OrderedDict

import dwi.dataset
import dwi.files
import dwi.paths
import dwi.util
from dwi.types import GleasonScore

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='increase verbosity')
    p.add_argument('--patients', required=True,
                   help='patient list file')
    p.add_argument('--mode', required=True,
                   help='imaging mode')
    p.add_argument('--param', type=int, default=0,
                   help='image parameter index')
    p.add_argument('--location',
                   help='lesion location')
    p.add_argument('--fig', required=True,
                   help='output figure file')
    return p.parse_args()


def plot(data, title=None, labels=None, path=None):
    if title:
        plt.title(title)
    if labels is None:
        labels = [str(x) for x in range(len(data))]
    labels = ['{} ({})'.format(l, len(d)) for l, d in zip(labels, data)]
    plt.boxplot(data, labels=labels)
    plt.tight_layout()
    if path:
        print('Plotting to {}...'.format(path))
        plt.savefig(path, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    lesions = list(dwi.dataset.iterlesions(args.patients))
    if args.location:
        lesions = [x for x in lesions if x[2].location == args.location]
    for _, _, l in lesions:
        l.label = sum(l.score > x for x in GleasonScore.THRESHOLDS_STANDARD)
    groups = OrderedDict([
        ('low', [(p, s, l) for p, s, l in lesions if l.label == 0]),
        ('int', [(p, s, l) for p, s, l in lesions if l.label == 1]),
        ('high', [(p, s, l) for p, s, l in lesions if l.label == 2]),
    ])

    print(len(lesions), [len(x) for x in groups.values()])
    assert sum(len(x) for x in groups.values()) == len(lesions)
    for k, v in groups.items():
        for c, s, l in v:
            print(k, c.num, s, l)

    data = []
    for i, lesions in enumerate(groups.values()):
        data.append([])
        for c, s, l in lesions:
            path = dwi.paths.roi_path(args.mode, 'lesion', c.num, s, l.index+1)
            if args.verbose:
                print('Reading', path)
            img, _ = dwi.files.read_pmap(path)
            img = img[..., args.param]
            img = img[np.isfinite(img)]
            data[i].append(np.mean(img))
    plot(data, title=args.mode, labels=list(groups.keys()), path=args.fig)


if __name__ == '__main__':
    main()
