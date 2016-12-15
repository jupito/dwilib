"""Shell commands for calculation tasks."""

from __future__ import absolute_import, division, print_function

from pathlib2 import Path

import dwi.util

DWILIB = Path.home() / 'src/dwilib/tools'


def paths_on_cmdline(paths):
    """Lay pathnames on command line."""
    return ' '.join('"{}"'.format(x) for x in paths)


def standardize_train_cmd(infiles, cfgpath, thresholding):
    """Standardize MRI images: training phase."""
    infiles = paths_on_cmdline(infiles)
    d = dict(prg=DWILIB/'standardize.py', o=cfgpath, i=infiles, t=thresholding)
    cmd = '{prg} -v --train {o} {i} --thresholding {t}'
    return cmd.format(**d)


def standardize_transform_cmd(cfgpath, inpath, outpath, mask=None):
    """Standardize MRI images: transform phase."""
    d = dict(prg=DWILIB/'standardize.py', c=cfgpath, i=inpath, o=outpath,
             m=mask)
    cmd = '{prg} -v --transform {c} {i} {o}'
    if mask:
        cmd += ' --mask {m}'
    return cmd.format(**d)


def get_texture_cmd(mode, inpath, method, winsize, slices, portion, outpath,
                    voxel, mask=None):
    d = dict(prg=DWILIB/'get_texture.py', m=mode, i=inpath, mask=mask,
             slices=slices, portion=portion, mth=method, ws=winsize,
             o=outpath, vx=voxel)
    cmd = ('{prg} -v'
           ' --mode {m}'
           ' --input {i}'
           ' --slices {slices} --portion {portion}'
           ' --method {mth} --winspec {ws} --voxel {vx}'
           ' --output {o}')
    if mask is not None:
        cmd += ' --mask {mask}'
    return cmd.format(**d)


def make_subregion_cmd(mask, subregion):
    d = dict(prg=DWILIB/'masktool.py', mask=mask, sr=subregion)
    cmd = '{prg} -i {mask} --pad 10 -s {sr}'
    return cmd.format(**d)


def select_voxels_cmd(inpath, outpath, mask=None, source_attrs=False,
                      astype=None, keepmasked=True):
    d = dict(prg=DWILIB/'select_voxels.py', i=inpath, o=outpath, m=mask,
             t=astype)
    cmd = '{prg} -i {i} -o {o}'
    if mask:
        cmd += ' -m {m}'
    if source_attrs:
        cmd += ' --source_attrs'
    if astype is not None:
        cmd += ' --astype {t}'
    if keepmasked:
        cmd += ' --keepmasked'
    return cmd.format(**d)


def mask_out_cmd(src, dst, mask):
    d = dict(prg=DWILIB/'mask_out_dicom.py', src=src, dst=dst, mask=mask)
    rm = 'rm -Rf {dst}'.format(**d)  # Remove destination image
    cp = 'cp -R --no-preserve=all {src} {dst}'.format(**d)  # Copy source
    mask = '{prg} --mask {mask} --image {dst}'.format(**d)  # Mask image
    return [rm, cp, mask]


def histogram_cmd(inpaths, figpath, param=0):
    d = dict(prg=DWILIB/'histogram.py', i=' '.join(inpaths), f=figpath,
             p=param)
    cmd = '{prg} -v --param {p} --input {i} --fig {f}'
    return cmd.format(**d)


def fit_cmd(infile, outfile, model, mask=None, mbb=None, params=None):
    d = dict(prg=DWILIB/'fit.py', i=infile, o=outfile, model=model, mask=mask,
             mbb=mbb, params=' '.join(str(x) for x in params))
    cmd = '{prg} -v --input {i} --output {o} --model {model}'
    if mask is not None:
        cmd += ' --mask {mask}'
    if mbb is not None:
        cmd += ' --mbb {mbb[0]} {mbb[1]} {mbb[2]}'
    if params is not None:
        cmd += ' --params {params}'
    return cmd.format(**d)


def arglist(args):
    return ' '.join(str(x) for x in args)


def grid_cmd(image, param, prostate, lesions, outpath, mbb=15, voxelsize=0.25,
             winsize=5, voxelspacing=None, lesiontypes=None,
             use_centroid=False, nanbg=False):
    prg = DWILIB / 'grid.py'
    lesions = arglist(lesions)
    voxelspacing = arglist(voxelspacing)
    lesiontypes = arglist(lesiontypes)
    cmd = ('{prg} -v --image {image} --prostate {prostate} --lesions {lesions}'
           ' --output {outpath}')
    if param is not None:
        cmd += ' --param {param}'
    if voxelsize is not None:
        cmd += ' --voxelsize {voxelsize}'
    if winsize is not None:
        cmd += ' --winsize {winsize}'
    if mbb is not None:
        cmd += ' --mbb {mbb}'
    if voxelspacing is not None:
        cmd += ' --voxelspacing {voxelspacing}'
    if lesiontypes is not None:
        cmd += ' --lesiontypes {lesiontypes}'
    if use_centroid:
        cmd += ' --use_centroid'
    if nanbg:
        cmd += ' --nanbg'
    return cmd.format(**locals())


def check_mask_overlap_cmd(container, other, fig):
    d = dict(prg=DWILIB/'check_mask_overlap.py', c=container, o=other, f=fig)
    cmd = '{prg} -v {c} {o} --fig {f}'
    return cmd.format(**d)
