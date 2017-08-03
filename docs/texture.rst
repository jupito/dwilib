Texture feature extraction
==========================

Texture feature extraction is done using `tools/get_texture.py`, which processes
a single image and texture method.

Handling multiple images can be automatized using `PyDoIt` with tasks `texture`
and `merge_textures`.

Example commands for task `texture`::

  doit run texture:DWI-Mono-ADCm_CA_all_0_71_1b_None_stats_all_all_median
  doit run texture:*_CA_all_0_*_*_*_median

The first command runs task `texture` for a target which is specified as
(mode, masktype, slices, portion, case, scan, lesion, method, winsize, voxel):

- mode DWI-Mono-ADCm,
- ROI specifier CA (cancer),
- slices: all (alternative: maxfirst for first maximum slice),
- portion: 0 (0/1 of window area must be inside mask, alternative: 1 for all),
- case: 71,
- scan id: 1b,
- lesion number beginning from 1, or None if not applicable,
- texture extraction method: stats,
- window size: all (side length or all),
- voxel specifier: median (alternatives: mean, all).

The second command uses wildcards expanded to all available combinations.

Example commands for task `merge_textures`::

  doit run merge_textures:DWI-Mono-ADCm_71_1b_None_CA_all_0_median
  doit run merge_textures:*_*_*_*_CA_all_0_median




ImageMode: modality-
TextureSpec: method-winsize
    raw-1
    gabor-11
ImageTarget(case, scan, lesion)
ROISpec(type, id)



Dataset structure and file locations
====================================

Program configuration:
    dwilib.cfg
Sample lists:
    patients/patients_{mode[0]}_{samplelist}.txt
    patients/patients_DWI_bothscansonly.txt
    patients/patients_T2w_bothscansonly.txt
Images:
    images/{mode[0:2]}/{case}-{scan}/{case}-{scan}_{mode[2]}.zip
    images/DWI-Mono/42-1a/42-1a_ADCm.zip
    images/T2w-std/42-1a.h5
Prostate masks:
    masks/{masktype}/{mode[0]}/{case}-{scan}.h5
    masks/prostate/DWI/42-1a.h5
    /mri/masks/prostate/T2w/42-1a.h5
Lesion masks:
    masks/lesion/{mode[0]/lesion{lesion}/{case}-{scan}.h5
    masks/lesion/DWI/lesion1/42-1a.h5
    masks/lesion/T2w/lesion1/42-1a.h5
ROI masks:
    masks/roi/{mode[0]}/{case}_{scan}_{roi}.h5
    masks/roi/DWI/42_1a_CA.h5
    masks/roi/DWI/42_1a_N.h5
