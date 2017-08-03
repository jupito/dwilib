"""Routines for handling patient lists."""

# TODO: Functions read_pmaps, read_pmap, grouping should be replaced with
# something better, they're still used by tools/{roc_auc,correlation}.py.

from functools import total_ordering


@total_ordering
class GleasonScore(object):
    """Gleason score is a two or three-value measure of prostate cancer
    severity.
    """
    # "Standard" thresholds for GS groups [low, intermediate, high]:
    # - low: maximum 3 anywhere;
    # - intermediate: 4 as secondary, or tertiary w/o 5;
    # - high: the rest.
    THRESHOLDS_STANDARD = ('3+3', '3+4')

    def __init__(self, score):
        """Intialize with a sequence or a string like '3+4+5' (third digit is
        optional).
        """
        if isinstance(score, str):
            s = score.split('+')
        elif isinstance(score, GleasonScore):
            s = score.score
        else:
            s = score
        s = tuple(int(x) for x in s)
        if len(s) == 2:
            s += (0,)  # Internal representation always has three digits.
        if len(s) != 3:
            raise ValueError('Invalid gleason score: {}'.format(score))
        self.score = s

    def __iter__(self):
        score = self.score
        if not score[-1]:
            score = score[0:-1]  # Drop trailing zero.
        return iter(score)

    def __repr__(self):
        return '+'.join(str(x) for x in iter(self))

    def __lt__(self, other):
        return self.score < GleasonScore(other).score

    def __eq__(self, other):
        return self.score == GleasonScore(other).score

    def __hash__(self):
        return hash(self.score)


class Lesion(object):
    """Lesion is a lump of cancer tissue."""
    def __init__(self, index, score, location):
        self.index = int(index)  # No. in patient.
        self.score = GleasonScore(score)  # Gleason score.
        self.location = str(location).lower()  # PZ or CZ.

    def __iter__(self):
        return iter([self.index, self.score, self.location])

    def __hash__(self):
        return hash(tuple(self))

    def __repr__(self):
        return repr(tuple(self))

    def __str__(self):
        return '({})'.format(','.join(str(x) for x in self))

    def __eq__(self, other):
        return (self.score, self.location) == (other.score, other.location)


@total_ordering
class Patient(object):
    """Patient case."""
    def __init__(self, num, name, scans, lesions):
        self.num = int(num)
        self.name = str(name).lower()
        self.scans = scans
        self.lesions = lesions
        self.score = lesions[0].score  # For backwards compatibility.

    def __repr__(self):
        return repr(self._astuple())

    def __hash__(self):
        return hash(self._astuple())

    def __eq__(self, other):
        return self._astuple() == other._astuple()

    def __lt__(self, other):
        return self._astuple() < other._astuple()

    def _astuple(self):
        return self.num, self.name, self.scans, self.lesions


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
