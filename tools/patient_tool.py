#!/usr/bin/env python2

"""Patient list tool."""

import argparse
import itertools

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
    """Label lesions according to score groups."""
    thresholds = map(dwi.patient.GleasonScore, thresholds)
    for p in patients:
        for l in p.lesions:
            l.label = sum(l.score > t for t in thresholds)
            l.patient = p

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


args = parse_args()
patients = dwi.files.read_patients_file(args.patients)
scores = dwi.patient.get_gleason_scores(patients)
thresholds = args.thresholds or scores
label_lesions(patients, thresholds)

#for p, s, l in dwi.patient.lesions(patients):
#    print p.num, l.index, l.score, l.label

all_lesions = list(itertools.chain(*(p.lesions for p in patients)))
n_labels = len({l.label for l in all_lesions})

label_groups = [[] for _ in range(n_labels)]
for lesion in all_lesions:
    label_groups[lesion.label].append(lesion)

group_sizes = [len(l) for l in label_groups]
label_patients(patients, group_sizes)

for p in patients:
    print p.num, p.label, [l.label for l in p.lesions]

for i in range(n_labels):
    print i, sum(1 for p in patients if p.label == i)

print 'Patients: {}, lesions: {}'.format(len(patients), len(all_lesions))
print 'Scores: {}: {}'.format(len(scores), scores)
print 'Group sizes: {}'.format(group_sizes)
