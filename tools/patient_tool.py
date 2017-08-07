#!/usr/bin/python3

"""Patient list tool."""

import argparse
import itertools
import random

import dwi.files
import dwi.patient
import dwi.util


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
                   help='be more verbose')
    p.add_argument('--patients',
                   help='patients file')
    p.add_argument('--thresholds', nargs='*', default=[],
                   help='classification thresholds (group maximums)')
    p.add_argument('--split', nargs=2, metavar='OUTFILE',
                   help='output train and test files')
    p.add_argument('--ratio', metavar='FLOAT', type=float, default=0.5,
                   help='split ratio')
    args = p.parse_args()
    return args


def label_patients(patients, group_sizes):
    """Label patients according to their lesion labels.

    Least-used labels are preferred in order to counter bias.
    """
    sorted_sizes = sorted((size, i) for i, size in enumerate(group_sizes))
    for p in patients:
        for _, i in sorted_sizes:
            if i in [l.label for l in p.lesions]:
                p.label = i
                break


def group_patients(patients):
    """Group patients by their label."""
    n_labels = len({p.label for p in patients})
    groups = [[p for p in patients if p.label == i] for i in range(n_labels)]
    return groups


def random_split(seq, ratio=0.5, surplus=0):
    """Randomly split sequncy into two.

    Parameter surplus defines the group where odd elements are put.
    """
    assert surplus == 0 or surplus == 1
    k = int(ratio * len(seq) + 0.5 * surplus)
    a = random.sample(seq, k)
    b = [x for x in seq if x not in a]
    return a, b


def main():
    args = parse_args()
    patients = dwi.files.read_patients_file(args.patients, include_lines=True)
    # Get all separate Gleason scores.
    scores = sorted({l.score for p in patients for l in p.lesions})
    thresholds = args.thresholds or scores
    dwi.patient.label_lesions(patients, thresholds)

    # For convenience, refer to patients in lesions.
    for p, l in ((p, l) for p in patients for l in p.lesions):
        l.patient = p

    # for p, s, l in dwi.dataset.iterlesions(patients):
    #     print(p.num, l.index, l.score, l.label)

    all_lesions = list(itertools.chain(*(p.lesions for p in patients)))
    n_labels = len({l.label for l in all_lesions})

    # Lesions grouped by label.
    label_groups = [[] for _ in range(n_labels)]
    for lesion in all_lesions:
        label_groups[lesion.label].append(lesion)

    # Number of lesions in each group.
    group_sizes = [len(l) for l in label_groups]
    label_patients(patients, group_sizes)

    # for p in patients:
    #     print(p.num, p.label, [l.label for l in p.lesions])

    # Split data into training set and test set.
    if args.split:
        patient_groups = group_patients(patients)
        train, test = [], []
        for i, g in enumerate(patient_groups):
            a, b = random_split(g, ratio=args.ratio, surplus=i % 2)
            train += a
            test += b
            print('Group: {i}, patients: {n}'.format(i=i, n=len(g)))
            print('  ', [p.num for p in sorted(a)])
            print('  ', [p.num for p in sorted(b)])
        for filename, seq in zip(args.split, [train, test]):
            d = dict(np=len(seq), nl=sum(len(p.lesions) for p in seq),
                     f=filename)
            print('Writing {np} patients, {nl} lesions to {f}...'.format(**d))
            with open(filename, 'w') as f:
                for p in sorted(seq):
                    f.write('{}\n'.format(p.line))

    print('Patients: {}, lesions: {}'.format(len(patients), len(all_lesions)))
    print('Scores: {}: {}'.format(len(scores), scores))
    print('Number of lesions in each group: {}'.format(group_sizes))

    print()
    print('Patients grouped by score group:')
    for i, g in enumerate(label_groups):
        pnums = sorted({l.patient.num for l in g})
        d = dict(i=i, n=len(pnums), p=pnums)
        print('{i}: {n} patients: {p}'.format(**d))

    print()
    print('Patients grouped by number of lesions:')
    min_lesions = min(len(p.lesions) for p in patients)
    max_lesions = max(len(p.lesions) for p in patients)
    for i in range(min_lesions, max_lesions+1):
        l = [p.num for p in patients if len(p.lesions) == i]
        print('{i} lesions: {n} patients: {l}'.format(i=i, n=len(l), l=l))

    print()
    print('Number of patients assigned to each group with bias counter:')
    for i in range(n_labels):
        print(i, sum(1 for p in patients if p.label == i))


if __name__ == '__main__':
    main()
