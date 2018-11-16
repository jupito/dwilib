Dataset structure and file locations
====================================

DoIt task definitions are in dodo.py. It's in dwilib main directory, but DoIt
expects it to be copied (or linked) to the current working directory. Other
option is to give the location as a command line argument.

Program configuration:
    dwilib.cfg
    - Example: https://github.com/jupito/dwilib/blob/master/examples/dwilib.cfg

Sample lists:
    patients/patients_{mode[0]}_{samplelist}.txt
    - Examples:
    patients/patients_DWI_bothscansonly.txt
    patients/patients_T2w_bothscansonly.txt

Images:
    images/{mode[0:2]}/{case}-{scan}/{case}-{scan}_{mode[2]}.zip
    - Examples:
    images/DWI-Mono/42-1a/42-1a_ADCm.zip
    images/T2w-std/42-1a.h5

Prostate masks:
    masks/{masktype}/{mode[0]}/{case}-{scan}.h5
    - Examples:
    masks/prostate/DWI/42-1a.h5
    /mri/masks/prostate/T2w/42-1a.h5

Lesion masks:
    masks/lesion/{mode[0]/lesion{lesion}/{case}-{scan}.h5
    - Examples:
    masks/lesion/DWI/lesion1/42-1a.h5
    masks/lesion/T2w/lesion1/42-1a.h5

ROI masks:
    masks/roi/{mode[0]}/{case}_{scan}_{roi}.h5
    - Examples:
    masks/roi/DWI/42_1a_CA.h5
    masks/roi/DWI/42_1a_N.h5

Texture calculations output:
    texture/...
