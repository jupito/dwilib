"""Routines for handling patient lists."""

# TODO: Some functions from dwi.compat should be replaced with something better
# here, they're still used by tools/{roc_auc,correlation}.py.

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


def keep_scan(patients, i):
    """Discard other scans except index i. NOTE: Changes the structure."""
    for p in patients:
        p.scans = [p.scans[i]]
