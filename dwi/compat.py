"""Obsolete code, kept for compatibility."""

from dwi.types import TextureSpec


def param_to_tspec(param):
    """Get partial TextureSpec from param string (only winsize and method!)."""
    winsize, name = param.split('-', 1)
    method = name.split('(', 1)[0]
    return TextureSpec(int(winsize), method, None)
