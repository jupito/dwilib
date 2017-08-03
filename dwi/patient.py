"""Routines for handling patient lists."""

# TODO: Functions read_pmaps, read_pmap, grouping should be replaced with
# something better, they're still used by tools/{roc_auc,correlation}.py.

from .types import GleasonScore


def label_lesions(patients, thresholds=None):
    """Label lesions according to score groups."""
    # Alternative: np.searchsorted(thresholds, [x.score for x in l])
    if thresholds is None:
        thresholds = GleasonScore.THRESHOLDS_STANDARD
    thresholds = [GleasonScore(x) for x in thresholds]
    lesions = (l for p in patients for l in p.lesions)
    for l in lesions:
        l.label = sum(l.score > x for x in thresholds)


def grouping(data):
    """Return different scores sorted, grouped scores, and their sample sizes.

    See read_pmaps()."""
    scores = [d['score'] for d in data]
    labels = [d['label'] for d in data]
    n_labels = max(labels) + 1
    groups = [[] for _ in range(n_labels)]
    for s, l in zip(scores, labels):
        groups[l].append(s)
    different_scores = sorted(set(scores))
    group_scores = [sorted(set(g)) for g in groups]
    group_sizes = [len(g) for g in groups]
    return different_scores, group_scores, group_sizes
