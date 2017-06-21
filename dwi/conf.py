"""Modifiable runtime configuration parameters."""

# TODO: Always read configuration to a argparse.Namespace object.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import shlex

from .types import Path
from . import util

log = logging.getLogger(__name__)

# Default runtime configuration parameters. Somewhat similar to matplotlib.
rcParamsDefault = {
    # 'cachedir': 'cache',
    'cachedir': str(Path('~/.cache/dwilib').expanduser()),
    'maxjobs': 0.9,
    'texture.methods': [
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
    'texture.winsizes.small': (3, 16, 2),  # DWI.
    # 'texture.winsizes.small': (11, 12, 2),  # DWI.
    'texture.winsizes.large': (3, 36, 4),  # T2, T2w.
    # 'texture.winsizes.large': (15, 36, 4),  # T2, T2w.
    'texture.avg': 'mean',  # Average result texture map (all, mean, median)?
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


class DefaultValueHelpFormatter(argparse.HelpFormatter):
    """A formatter that appends possible default value to argument helptext."""
    def _expand_help(self, action):
        s = super()._expand_help(action)
        default = getattr(action, 'default', None)
        if default is None or default in [False, argparse.SUPPRESS]:
            return s
        return '{} (default: {})'.format(s, repr(default))


class MyArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser. Added `add` as a shortcut to `add_argument`; set
    `fromfile_prefix_chars` by default; better `convert_arg_line_to_args()`;
    added `parse_from_files()`.
    """
    add = argparse.ArgumentParser.add_argument

    def __init__(self, **kwargs):
        kwargs.setdefault('fromfile_prefix_chars', '@')
        super().__init__(**kwargs)

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
    p = MyArgumentParser(add_help=False)
    p.add('-v', '--verbose', action='count', default=0,
          help='increase verbosity')
    p.add('--logfile', type=expanded_path, help='log file')
    p.add('--loglevel', default='WARNING', help='log level name')
    p.add('-j', '--maxjobs', type=float, default=0.9,
          help=('maximum number of simultaneous jobs '
                '(absolute, portion of CPU count, or negative count)'))
    p.add('-s', '--samplelist', default='all', help='samplelist identifier')
    p.add('--texture_methods', nargs='+', help='texture methods')
    return p


def get_parser(formatter_class=DefaultValueHelpFormatter, **kwargs):
    """Get an argument parser with the usual standard arguments ready."""
    parents = [get_config_parser()]
    p = MyArgumentParser(parents=parents, formatter_class=formatter_class,
                         **kwargs)
    return p


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

    if namespace.texture_methods is not None:
        rcParams['texture.methods'] = namespace.texture_methods

    if parser is not None:
        parser.parse_args(namespace=namespace)
    init_logging(namespace)

    # TODO: Under construction.
    for k, v in vars(namespace).items():
        k = k.translate(str.maketrans('_', '.'))  # Change '_' to '.'
        rcParams[k] = v

    log.debug('Parsed args: %s', namespace)
    it = ('\n\t{k}: {v}'.format(k=k, v=v) for k, v in
          sorted(vars(namespace).items()))
    log.debug('Parsed config: ...%s', ''.join(it))

    return namespace
