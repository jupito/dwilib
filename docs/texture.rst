Texture feature extraction
==========================

Texture feature extraction is done using `tools/get_texture.py`, which processes
a single image and texture method.

Handling multiple images can be automatized using `PyDoIt` with tasks `texture`
and `merge_textures`.

Example commands for task `texture`::

  doit run texture:DWI-Mono-ADCm_CA_all_0_71_1b_None_stats_all_median
  doit run texture:*_CA_all_0_*_*_*_gabor_9_mean

The first command runs task `texture` for a target which is specified as
(mode, masktype, slices, portion, case, scan, lesion, method, winsize, voxel):

- mode DWI-Mono-ADCm,
- ROI specifier CA (cancer),
- slices: all (alternative: maxfirst for first maximum slice),
- portion: 0 (0/1 of window area must be inside mask, alternative: 1 for all),
- case: 71,
- scan id: 1b,
- lesion number beginning from 1, or None if not applicable,
- texture extraction method: stats, gabor, ...
- window size: all (side length, all, mbb),
- voxel specifier: median (all, mean, median).

The second command uses wildcards expanded to all available combinations.

Example commands for task `merge_textures`::

  doit run merge_textures:DWI-Mono-ADCm_71_1b_None_CA_all_0_median
  doit run merge_textures:*_*_*_*_CA_all_0_median




ImageMode: modality-model-param
TextureSpec: method-winsize
    raw-1
    gabor-11
ImageTarget: case-scan-lesion
ROISpec: type-id



