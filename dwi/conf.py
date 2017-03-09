"""Modifiable runtime configuration parameters."""

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os.path
import shlex

from dwi.files import Path
import dwi.util

# Default runtime configuration parameters. Somewhat similar to matplotlib.
rcParamsDefault = {
    'texture.methods': [
        'stats',
        # 'haralick',
        # 'moment',
        # 'haralick_mbb',
        'glcm',
        # 'glcm_mbb',
        'lbp',
        'hog',
        'gabor',
        'haar',
        'hu',
        'zernike',
        'sobel',
        # 'stats_all',
        ],
    'texture.winsizes.large': (3, 36, 4),  # T2, T2w.
    'texture.winsizes.small': (3, 16, 2),  # DWI.
    'texture.avg': False,  # Boolean: average result texture map?
    'texture.path': None,  # Write result directly to disk, if string.
    'texture.dtype': 'float32',  # Output texture map type.
    'texture.glcm.names': ('contrast', 'dissimilarity', 'homogeneity',
                           'energy', 'correlation', 'ASM'),
    'texture.glcm.distances': (1, 2, 3, 4),  # GLCM pixel distances.
    'texture.gabor.orientations': 4,  # Number of orientations.
    # 'texture.gabor.orientations': 6,  # Number of orientations.
    'texture.gabor.sigmas': (1, 2, 3),
    # 'texture.gabor.sigmas': (None,),
    'texture.gabor.freqs': (0.1, 0.2, 0.3, 0.4, 0.5),
    'texture.lbp.neighbours': 8,  # Number of neighbours.
    'texture.zernike.degree': 8,  # Maximum degree.
    'texture.haar.levels': 4,  # Numer of levels.
    'texture.hog.orientations': 1,  # Numer of orientations.
    }
rcParams = dict(rcParamsDefault)


def rcdefaults():
    """Restore default rc params."""
    rcParams.update(rcParamsDefault)


def get_config_paths():
    """Return existing default configuration files."""
    dirnames = ['/etc/dwilib', '~/.config/dwilib', '.']
    # Py3.4 pathlib has no expanduser().
    dirnames = [os.path.expanduser(x) for x in dirnames]
    filename = 'dwilib.cfg'
    # paths = [Path(x).expanduser() / filename for x in dirnames]
    paths = [Path(x) / filename for x in dirnames]
    paths = [x for x in paths if x.exists()]
    return paths


def parse_config(parser):
    """Parse configuration files."""
    prefix = parser.fromfile_prefix_chars[0]
    args = ['{}{}'.format(prefix, x) for x in get_config_paths()]
    namespace, _ = parser.parse_known_args(args)
    return namespace


def convert_arg_line_to_args(line):
    """A better replacement for ArgumentParser."""
    return shlex.split(line, comments=True)


def get_parser(**kwargs):
    """Get an argument parser with the usual standard arguments ready. Function
    'add' is added for convenience.
    """
    p = argparse.ArgumentParser(fromfile_prefix_chars='@', **kwargs)
    p.convert_arg_line_to_args = convert_arg_line_to_args
    p.add = p.add_argument
    p.add('-v', '--verbose', action='count', default=0,
          help='increase verbosity')
    p.add('--logfile', help='log file')
    p.add('--loglevel', default='WARNING', help='log level name')
    return p


def init_logging(args):
    """Initialize logging."""
    logging.basicConfig(filename=args.logfile,
                        level=dwi.util.get_loglevel(args.loglevel))


def parse_args(parser):
    """Parse args and configuration as well."""
    namespace = parse_config(parser)
    args = parser.parse_args(namespace=namespace)
    dwi.conf.init_logging(args)
    return args
