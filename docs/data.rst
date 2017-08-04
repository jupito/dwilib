Dataset structure and file locations
====================================

Program configuration:
    dwilib.cfg

DoIt task definitions:
    dodo.py (must be in current working directory or specified on command line)

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

Texture calculations:
    texture/...
