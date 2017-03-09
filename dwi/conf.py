"""Modifiable runtime configuration parameters."""

from __future__ import absolute_import, division, print_function
import argparse
import logging
import shlex

from dwi.files import Path
import dwi.util

# Default runtime configuration parameters. Somewhat similar to matplotlib.
rcParamsDefault = {
    'texture.methods': [
        'raw',
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
    # # Py3.4 pathlib has no expanduser().
    # dirnames = [os.path.expanduser(x) for x in dirnames]
    filename = 'dwilib.cfg'
    paths = [Path(x).expanduser() / filename for x in dirnames]
    paths = [x for x in paths if x.exists()]
    return paths


def parse_config(parser):
    """Parse configuration files."""
    # prefix = parser.fromfile_prefix_chars[0]
    # args = ['{}{}'.format(prefix, x) for x in get_config_paths()]
    # namespace, _ = parser.parse_known_args(args)
    namespace, _ = parser.parse_from_files(get_config_paths())
    return namespace


def expanded_path(*args, **kwargs):
    """Automatically expanded Path. Useful as an argparse type from file."""
    return Path(*args, **kwargs).expanduser()


class MyArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser. Added `add` as a shortcut to `add_argument`; set
    `fromfile_prefix_chars` by default; better `convert_arg_line_to_args()`;
    added `parse_from_files()`.
    """
    add = argparse.ArgumentParser.add_argument

    def __init__(self, **kwargs):
        kwargs.setdefault('fromfile_prefix_chars', '@')
        super(self.__class__, self).__init__(**kwargs)

    def convert_arg_line_to_args(self, line):
        """Fancier file reading."""
        return shlex.split(line, comments=True)

    def parse_from_files(self, paths):
        """Parse known arguments from files."""
        prefix = self.fromfile_prefix_chars[0]
        args = ['{}{}'.format(prefix, x) for x in paths]
        return self.parse_known_args(args)

    # raise_on_error = True
    # def error(self, message):
    #     if self.raise_on_error:
    #         raise ValueError(message)
    #     super(self.__class__, self).error(message)


def get_parser(**kwargs):
    """Get an argument parser with the usual standard arguments ready."""
    p = MyArgumentParser(**kwargs)
    p.add('-v', '--verbose', action='count', default=0,
          help='increase verbosity')
    p.add('--logfile', type=expanded_path, help='log file')
    p.add('--loglevel', default='WARNING', help='log level name')
    return p


def init_logging(args):
    """Initialize logging."""
    d = {}
    if args.logfile is not None:
        d['filename'] = str(args.logfile)
    if args.loglevel is not None:
        d['level'] = dwi.util.get_loglevel(args.loglevel)
    logging.basicConfig(**d)


def parse_args(parser):
    """Parse args and configuration as well."""
    namespace = parse_config(parser)
    args = parser.parse_args(namespace=namespace)
    dwi.conf.init_logging(args)
    return args
