Instructions for using dwilib
=============================

Fitting models to data
----------------------

Program pmap.py is used for fitting the diffusion models to imaging data. It
generates parametric maps, hence the name. Invoking "pmap.py --help" lists
possible arguments for the program. Use -l to list all available models.

    usage: pmap.py [-h] [-v] [-l] [-a] [-s I I I I I I] [-m MODEL [MODEL ...]]
                   [-i FILENAME [FILENAME ...]] [-d PATHNAME [PATHNAME ...]]
                   [-o FILENAME]

    Produce parametric maps by fitting one or more diffusion models to imaging
    data. Multiple input images can be provided in ASCII format. Single input
    image can be provided as a group of DICOM files. Output is written in ASCII
    files named by input and model.

    optional arguments:
      -h, --help            show this help message and exit
      -v, --verbose         increase verbosity
      -l, --listmodels      list available models
      -a, --average         average input voxels into one
      -s I I I I I I, --subwindow I I I I I I
                            use subwindow (specified by 6 one-based indices)
      -m MODEL [MODEL ...], --models MODEL [MODEL ...]
                            models to use
      -i FILENAME [FILENAME ...], --input FILENAME [FILENAME ...]
                            input ASCII files
      -d PATHNAME [PATHNAME ...], --dicom PATHNAME [PATHNAME ...]
                            input DICOM files or directories
      -o FILENAME, --output FILENAME
                            output file (for single model only)

Subwindow specification (-s) is a list of six one-based index numbers in the
following order: first included slice, next excluded slice, first included row,
next excluded row, first included column, next excluded column. For example,
subwindow "9 10 70 120 80 140" includes the 9th slice with an area of 50x60
voxels starting from the 70th row and 80th column.

Output files are in simple ASCII format. The header contains information about
the subwindow, b-values, parameter names, etc. After that, each line represents
one voxel with its fitted parameter values separated by space, and ending in
the root mean square error (RMSE).

Examples:

    pmap.py -s 11 12 85 135 80 140 -m MonoN KurtN -d 10_1a/DICOM/

fits the normalized Monoexponential ADC and Kurtosis models to a one-slice
subwindow of a DICOM image from directory "10_1a/DICOM", outputting each
parametric map to its on file.

    pmap.py -s 11 12 85 135 80 140 -m normalized -d 10_1a/DICOM/

fits all normalized models.

    pmap.py -m SiN -d 10_1a/DICOM/ -o out.txt

generates the normalized signal intensity curves for the whole image into file
out.txt.
