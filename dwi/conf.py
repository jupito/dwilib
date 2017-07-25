"""Modifiable runtime configuration parameters."""

from __future__ import absolute_import, division, print_function
import argparse
import logging
from pprint import pformat
import shlex

from .types import Path
from . import util

log = logging.getLogger(__name__)


def expanded_path(*args, **kwargs):
    """Automatically expanded Path. Useful as an argparse type from file."""
    return Path(*args, **kwargs).expanduser()


def get_config_paths():
    """Return existing default configuration files."""
    dirnames = ['/etc/dwilib', '~/.config/dwilib', '.']
    filename = 'dwilib.cfg'
    paths = [expanded_path(x) / filename for x in dirnames]
    paths = [x for x in paths if x.exists()]
    return paths


def parse_config(parser):
    """Parse configuration files."""
    # prefix = parser.fromfile_prefix_chars[0]
    # args = ['{}{}'.format(prefix, x) for x in get_config_paths()]
    # namespace, _ = parser.parse_known_args(args)
    namespace, _ = parser.parse_from_files(get_config_paths())
    return namespace


class DefaultValueHelpFormatter(argparse.HelpFormatter):
    """A formatter that appends possible default value to argument helptext."""
    def _expand_help(self, action):
        s = super()._expand_help(action)
        default = getattr(action, 'default', None)
        if default is None or default in [False, argparse.SUPPRESS]:
            return s
        return '{} (default: {})'.format(s, repr(default))


class FileArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that is better for reading arguments from files.

    Set `fromfile_prefix_chars` to `@` by default; better
    `convert_arg_line_to_args()`; added `parse_from_files()`; added `add` as a
    shortcut to `add_argument`.
    """
    add = argparse.ArgumentParser.add_argument

    def __init__(self, fromfile_prefix_chars='@', **kwargs):
        super().__init__(fromfile_prefix_chars=fromfile_prefix_chars, **kwargs)

    def convert_arg_line_to_args(self, arg_line):
        """Fancier file reading."""
        return shlex.split(arg_line, comments=True)

    def parse_from_files(self, paths):
        """Parse known arguments from files."""
        prefix = self.fromfile_prefix_chars[0]
        args = ['{}{}'.format(prefix, x) for x in paths]
        return self.parse_known_args(args)

    # raise_on_error = True
    # def error(self, message):
    #     if self.raise_on_error:
    #         raise ValueError(message)
    #     super().error(message)


def get_config_parser():
    """Get configuration parser."""
    p = FileArgumentParser(add_help=False)
    p.add('-v', '--verbose', action='count', default=0,
          help='increase verbosity')
    p.add('--logfile', type=expanded_path, help='log file')
    p.add('--loglevel', default='WARNING', help='log level name')

    p.add('--cachedir', type=expanded_path, default='~/.cache/dwilib')
    p.add('--maxjobs', type=float, default=0.9,
          help=('maximum number of simultaneous jobs '
                '(absolute, portion of CPU count, or negative count)'))
    p.add('--texture_methods', nargs='+',
          default=[
              'raw',
              'stats',
              # 'haralick',
              # 'moment',
              # 'haralick_mbb',
              'glcm',
              'glcm_mbb',
              # 'lbp',
              # 'hog',
              'gabor',
              # 'haar',
              'hu',
              'zernike',
              # 'sobel',
              'stats_mbb',
              'stats_all',
              ],
          help='texture methods')
    p.add('--texture_winsizes_small', nargs=3, type=int,
          default=(3, 16, 2),
          # default=(11, 12, 2)
          help='window sizes for DWI')
    p.add('--texture_winsizes_large', nargs=3, type=int,
          default=(3, 36, 4),
          # default=(15, 36, 4)
          help='window sizes for T2, T2w')
    p.add('--texture_avg',
          default='median',
          help='average result texture map (all, mean, median)?')
    p.add('--texture_path', type=expanded_path,
          default=None,
          help='write result directly to disk, if string')
    p.add('--texture_dtype',
          default='float32',
          help='output texture map type')
    p.add('--texture_glcm_names', nargs='+',
          default=('contrast', 'dissimilarity', 'homogeneity', 'energy',
                   'correlation', 'ASM'),
          help='GLCM features to calculate')
    p.add('--texture_glcm_distances', nargs='+', type=int,
          default=(1, 2, 3, 4),
          help='GLCM pixel distances')
    p.add('--texture_gabor_orientations', type=int,
          default=4,
          # default=6
          help='number of orientations')
    p.add('--texture_gabor_sigmas', nargs='+', type=float,
          default=(1, 2, 3),
          # default=(None,)
          help='sigmas')
    p.add('--texture_gabor_freqs', nargs='+', type=float,
          default=(0.1, 0.2, 0.3, 0.4, 0.5),
          help='frequencies')
    p.add('--texture_lbp_neighbours', type=int,
          default=8,
          help='number of neighbours')
    p.add('--texture_zernike_degree', type=int,
          default=8,
          help='maximum degree')
    p.add('--texture_haar_levels', type=int,
          default=4,
          help='number of levels')
    p.add('--texture_hog_orientations', type=int,
          default=1,
          help='number of orientations')
    return p


def get_basic_parser():
    """Get basic parser."""
    p = FileArgumentParser(add_help=False)
    p.add('-v', '--verbose', action='count', default=0,
          help='increase verbosity')
    p.add('--logfile', type=expanded_path, help='log file')
    p.add('--loglevel', default='WARNING', help='log level name')
    return p


def get_parser(formatter_class=DefaultValueHelpFormatter, **kwargs):
    """Get an argument parser with the usual standard arguments ready."""
    parents = [get_basic_parser()]
    return FileArgumentParser(parents=parents, formatter_class=formatter_class,
                              **kwargs)


def init_logging(args):
    """Initialize logging."""
    d = {}
    if args.logfile is not None:
        d['filename'] = str(args.logfile)
    if args.loglevel is not None:
        d['level'] = util.get_loglevel(args.loglevel)
    logging.basicConfig(**d)


def parse_args(parser=None):
    """Parse args and configuration as well."""
    config_parser = get_config_parser()
    namespace = parse_config(config_parser)

    if parser is not None:
        parser.parse_args(namespace=namespace)
    init_logging(namespace)

    log.debug('Parsed args: %s', pformat(vars(namespace)))

    return namespace


rcParams = parse_args()
log.debug('Parsed config: %s', pformat(rcParams))
