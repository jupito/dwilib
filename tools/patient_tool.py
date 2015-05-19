#!/usr/bin/env python2

"""Patient list tool."""

import argparse
import collections

import dwi.files
import dwi.patient
import dwi.util

def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verbose', '-v', action='count',
            help='be more verbose')
    p.add_argument('--patients', default='patients.txt',
            help='patients file')
    p.add_argument('--thresholds', nargs='*', default=[],
            help='classification thresholds (group maximums)')
    args = p.parse_args()
    return args

def label_lesions(patients, thresholds):
    thresholds = map(dwi.patient.GleasonScore, thresholds)
    for p in patients:
        for l in p.lesions:
            l.label = sum(l.score > t for t in thresholds)
            l.patient = p


# Collect all parameters.
args = parse_args()
patients = dwi.files.read_patients_file(args.patients)
scores = dwi.patient.get_gleason_scores(patients)
thresholds = args.thresholds or scores
label_lesions(patients, thresholds)

print scores
#for p in patients:
#    print p

#for p, s, l in dwi.patient.lesions(patients):
#    print p.num, l.index, l.score, l.label

lesions = reduce(lambda x, y: x+y, (p.lesions for p in patients))
max_label = max(l.label for l in lesions)

label_groups = collections.defaultdict(list)
for lesion in lesions:
    label_groups[lesion.label].append(lesion)

for label in sorted(label_groups.keys()):
    l = label_groups[label]
    print label, len(l)
    for lesion in l:
        print lesion.patient.num, lesion
