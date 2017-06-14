"""Miscellaneous types."""

from collections import namedtuple

from .files import Path, PurePath
from .patient import GleasonScore, Lesion, Patient
from .util import ImageMode


TextureSpec = namedtuple('TextureSpec', ['winsize', 'method', 'feature'])

__all__ = list(n for n in globals() if n[:1] != '_')
