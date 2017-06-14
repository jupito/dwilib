"""Miscellaneous types."""

from collections import namedtuple

from dwi.files import Path, PurePath
from dwi.patient import GleasonScore, Lesion, Patient
from dwi.util import ImageMode


TextureSpec = namedtuple('TextureSpec', ['winsize', 'method', 'feature'])

__all__ = list(n for n in globals() if n[:1] != '_')
