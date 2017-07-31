"""Miscellaneous types."""

from collections import namedtuple
from functools import total_ordering

# With older Python version, pathlib2 might be preferred.
try:
    from pathlib2 import Path, PurePath
except ImportError:
    from pathlib import Path, PurePath

from . import util
from .patient import GleasonScore, Lesion, Patient


@total_ordering
class ImageMode(object):
    """Image mode identifier."""
    def __init__(self, value, sep='-'):
        """Initialize with a string or a sequence."""
        if util.isstring(value):
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

    def __hash__(self):
        return hash(tuple(self))

    # def __add__(self, other):
    #     """Append a component."""
    #     return self.__class__(self.value + (other,))

    # def __sub__(self, other):
    #     """Remove a tailing component."""
    #     v = self.value
    #     if v[-1] == other:
    #         v = v[:-1]
    #     return self.__class__(v)


def _fmt_seq(seq):
    return '-'.join(str(x) for x in seq if x is not None)


def namedtuple_fmt(*args, **kwargs):
    t = namedtuple(*args, **kwargs)
    t.__str__ = _fmt_seq
    return t


TextureSpec = namedtuple_fmt('TextureSpec', ['winsize', 'method', 'feature'])
ImageTarget = namedtuple_fmt('ImageTarget', ['case', 'scan', 'lesion'])
ROISpec = namedtuple_fmt('ROISpec', ['type', 'id'])
AlgParams = namedtuple_fmt('AlgParams', ['depthmin', 'depthmax',
                                         'sidemin', 'sidemax', 'nrois'])

__all__ = list(n for n in globals() if n[:1] != '_')
