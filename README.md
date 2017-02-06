dwilib
======

Tools in development to support analysis of Diffusion-Weighted Imaging (DWI)
data, with focus on prostate cancer.

This software is being developed as part of a research project at the [Magnetic
Resonance Imaging Research Center](http://mrc.utu.fi/), Turku, Finland.

Note: This is very much under construction. The program code is written on
demand basis, for custom needs at in-house projects.


Features
--------
- Read input data as DICOM, or in custom ASCII or MATLAB formats
- Perform model fitting (Monoexponential ADC, Kurtosis, Stretched exponential,
  Biexponential)
- Calculate correlation with Gleason score
- Calculate and compare diagnostic ROC AUCs
- Calculate reproducibility measures
- Plotting schemes
- Viewer for multi-slice, multi-b-value DWI DICOM files (uses the Matplotlib GUI
  widget)
- Build tool for automated pipelining of data processing tasks


Requirements
------------
- Python 2.7
- pathlib2
- NumPy
- SciPy
- [Scikit-Learn](http://scikit-learn.org/)
- [leastsqbound-scipy](https://github.com/jjhelmus/leastsqbound-scipy) (if
  fitting)
- [Matplotlib](http://matplotlib.org/) (if plotting)
- [Pydicom](https://code.google.com/p/pydicom/) (if reading DICOM files)
- [scikit-image](http://scikit-image.org/) (if doing texture analysis)
- [Mahotas](http://luispedro.org/software/mahotas/) (if doing texture analysis)
- [DoIt](http://pydoit.org/) (if using the build tool)


Todo or In Progress
-------------------
- Improved documentation
- Improved build tool
- Autonomous tumor delineation/ROI placement
- Regression classification
