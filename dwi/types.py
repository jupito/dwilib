"""Miscellaneous types."""

from collections import namedtuple
from functools import total_ordering

# With older Python version, pathlib2 might be preferred.
try:
    from pathlib2 import Path, PurePath
except ImportError:
    from pathlib import Path, PurePath  # noqa


@total_ordering
class ImageMode(object):
    """Image mode identifier."""
    def __init__(self, value, sep='-'):
        """Initialize with a string or a sequence."""
        if isinstance(value, str):
            value = value.split(sep)
        self.value = tuple(value)
        self.sep = sep

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key):
        return self.__class__(self.value[key])

    def __len__(self):
        return len(self.value)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, ', '.join(self))

    def __str__(self):
        return self.sep.join(iter(self))

    def __lt__(self, other):
        return tuple(self) < tuple(ImageMode(other))

    def __eq__(self, other):
        return tuple(self) == tuple(ImageMode(other))

    # def __hash__(self):
    #     return hash(tuple(self))


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
        return hash(tuple(self))


class Lesion(object):
    """Lesion is a lump of cancer tissue."""
    def __init__(self, index, score, location):
        self.index = int(index)  # No. in patient.
        self.score = GleasonScore(score)  # Gleason score.
        self.location = str(location).lower()  # PZ or CZ.

    def __iter__(self):
        return iter([self.index, self.score, self.location])

    # def __hash__(self):
    #     return hash(tuple(self))

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

    # def __hash__(self):
    #     return hash(self._astuple())

    def __eq__(self, other):
        return self._astuple() == other._astuple()

    def __lt__(self, other):
        return self._astuple() < other._astuple()

    def _astuple(self):
        return self.num, self.name, self.scans, self.lesions


def namedtuple_fmt(*args, **kwargs):
    """Namedtuple with added formatting and parsing."""
    sep = kwargs.pop('sep', '-')

    def _fmt_seq(seq):
        if not seq:
            return ''
        if seq[-1] is None:
            return _fmt_seq(seq[:-1])
        return sep.join(str(x) for x in seq)

    @classmethod
    def _parse(cls, s):
        return cls(*(x or None for x in s.split(sep)))

    t = namedtuple(*args, **kwargs)
    t.__str__ = _fmt_seq
    t._parse = _parse
    return t


# NOTE: Experimental.
# TODO: See
# https://stackoverflow.com/questions/3223236/creating-a-namedtuple-with-a-custom-hash-function
ImageMode_NT = namedtuple_fmt('ImageMode_NT', ['modality', 'model', 'param'])
GleasonScore_NT = namedtuple_fmt('GleasonScore_NT', ['primary', 'secondary',
                                                     'tertiary'], sep='+')

ImageTarget = namedtuple_fmt('ImageTarget', ['case', 'scan', 'lesion'])
TextureSpec = namedtuple_fmt('TextureSpec', ['method', 'winsize', 'feature'])
ROISpec = namedtuple_fmt('ROISpec', ['type', 'id'])
AlgParams = namedtuple_fmt('AlgParams', ['depthmin', 'depthmax',
                                         'sidemin', 'sidemax', 'nrois'])

__all__ = list(n for n in globals() if n[:1] != '_')
