"""Shell commands for calculation tasks."""

from __future__ import absolute_import, division, print_function

import dwi.util


DWILIB = '~/src/dwilib/tools'


def cmdline(*positionals, **options):
    """Construct a shell command string."""
    l = list(positionals)
    for k, v in sorted(options.iteritems()):
        if v is not None:
            k = str(k)
            dashes = '--' if len(k) > 1 else '-'
            l.append(dashes + k)
            if dwi.util.iterable(v) and not isinstance(v, basestring):
                l.extend(v)
            else:
                l.append(v)
    return ' '.join(str(x) for x in l)


def paths_on_cmdline(paths):
    """Lay pathnames on command line."""
    return ' '.join('"{}"'.format(x) for x in paths)


def standardize_train_cmd(infiles, cfgpath, thresholding):
    """Standardize MRI images: training phase."""
    infiles = paths_on_cmdline(infiles)
    cmd = '{prg} -v --train {o} {i} --thresholding {t}'
    return cmd.format(prg=DWILIB+'/standardize.py', o=cfgpath, i=infiles,
                      t=thresholding)


def standardize_transform_cmd(cfgpath, inpath, outpath, mask=None):
    """Standardize MRI images: transform phase."""
    cmd = '{prg} -v --transform {c} {i} {o}'
    if mask:
        cmd += ' --mask {m}'
    return cmd.format(prg=DWILIB+'/standardize.py', c=cfgpath, i=inpath,
                      o=outpath, m=mask)


def get_texture_cmd(mode, inpath, method, winsize, slices, portion, mask,
                    outpath, voxel):
    d = dict(prg=DWILIB+'/get_texture.py', m=mode, i=inpath, mask=mask,
             slices=slices, portion=portion, mth=method, ws=winsize,
             o=outpath, vx=voxel)
    cmd = ('{prg} -v'
           ' --mode {m}'
           ' --input {i} --mask {mask}'
           ' --slices {slices} --portion {portion}'
           ' --method {mth} --winspec {ws} --voxel {vx}'
           ' --output {o}')
    return cmd.format(**d)


# def find_roi_cmd(mode, case, scan, algparams, outmask, outfig):
#     d = dict(prg=DWILIB+'/find_roi.py', m=mode,
#              slf=samplelist_path(mode, SAMPLELIST), pd=pmap_path(mode),
#              srd=subregion_path(mode), c=case, s=scan,
#              ap=' '.join(algparams),
#              outmask=outmask, outfig=outfig)
#     return ('{prg} --patients {slf} --pmapdir {pd} --subregiondir {srd} '
#             '--param {m[2]} --cases {c} --scans {s} --algparams {ap} '
#             '--outmask {outmask} --outfig {outfig}'.format(**d))


def make_subregion_cmd(mask, subregion):
    cmd = '{prg} -i {mask} --pad 10 -s {sr}'
    return cmd.format(prg=DWILIB+'/masktool.py', mask=mask, sr=subregion)


def select_voxels_cmd(inpath, outpath, mask=None, source_attrs=False,
                      astype=None, keepmasked=True):
    cmd = '{prg} -i {i} -o {o}'
    if mask:
        cmd += ' -m {m}'
    if source_attrs:
        cmd += ' --source_attrs'
    if astype is not None:
        cmd += ' --astype {t}'
    if keepmasked:
        cmd += ' --keepmasked'
    return cmd.format(prg=DWILIB+'/select_voxels.py', i=inpath, o=outpath,
                      m=mask, t=astype)


# def auc_cmd(mode, threshold, algparams, outfile):
#     d = dict(prg=DWILIB+'/roc_auc.py', m=mode,
#              slf=samplelist_path(mode, SAMPLELIST), t=threshold,
#              i=roi_path(mode, 'auto', algparams=algparams),
#              ap_='_'.join(algparams), o=outfile)
#     return (r'echo `{prg} --patients {slf} --threshold {t} --voxel mean'
#             '--autoflip --pmapdir {i}` {ap_} >> {o}'.format(**d))


# def correlation_cmd(mode, thresholds, algparams, outfile):
#     d = dict(prg=DWILIB+'/correlation.py', m=mode,
#              slf=samplelist_path(mode, SAMPLELIST), t=thresholds,
#              i=roi_path(mode, 'auto', algparams=algparams),
#              ap_='_'.join(algparams), o=outfile)
#     return (r'echo `{prg} --patients {slf} --thresholds {t} --voxel mean'
#             '--pmapdir {i}` {ap_} >> {o}'.format(**d))


def mask_out_cmd(src, dst, mask):
    d = dict(prg=DWILIB+'/mask_out_dicom.py', src=src, dst=dst, mask=mask)
    rm = 'rm -Rf {dst}'.format(**d)  # Remove destination image
    cp = 'cp -R --no-preserve=all {src} {dst}'.format(**d)  # Copy source
    mask = '{prg} --mask {mask} --image {dst}'.format(**d)  # Mask image
    return [rm, cp, mask]


def histogram_cmd(inpaths, figpath):
    return '{prg} -v --param 0 --input {i} --fig {f}'.format(
        prg=DWILIB+'/histogram.py', i=' '.join(inpaths), f=figpath)


def fit_cmd(infile, outfile, model, mask=None, mbb=None, params=None):
    s = '{prg} -v --input {i} --output {o} --model {model}'
    if mask is not None:
        s += ' --mask {mask}'
    if mbb is not None:
        s += ' --mbb {mbb[0]} {mbb[1]} {mbb[2]}'
    if params is not None:
        s += ' --params {params}'
    return s.format(prg=DWILIB+'/fit.py', i=infile, o=outfile, model=model,
                    mask=mask, mbb=mbb,
                    params=' '.join(str(x) for x in params))


def grid_cmd(image, param, prostate, lesions, outpath, mbb=15, voxelsize=0.25,
             winsize=5, voxelspacing=None):
    d = dict(v=(), mbb=mbb, voxelsize=voxelsize, winsize=winsize, image=image,
             prostate=prostate, lesions=lesions, output=outpath)
    if param is not None:
        d.update(param=param)
    if voxelspacing is not None:
        d.update(voxelspacing=voxelspacing)
    return cmdline(DWILIB+'/grid.py', **d)


def check_mask_overlap_cmd(container, other, fig):
    s = '{prg} -v {c} {o} --fig {f}'
    return s.format(prg=DWILIB+'/check_mask_overlap.py', c=container, o=other,
                    f=fig)
