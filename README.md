dwilib
======

Tools in development to support analysis of Diffusion-weighted imaging (DWI)
data, with focus on prostate cancer.

Input data can be provided in DICOM or custom ASCII or MATLAB format.

This software is being developed as part of a research project at [University
of Turku](http://www.utu.fi/).

Features
--------
- Perform model fitting (Monoexponential ADC, Kurtosis, Stretched exponential,
  Biexponential)
- Calculate correlation by Gleason scores
- Calculate and compare diagnostic ROC AUCs
- Calculate reproducibility measures
- Plotting schemes
- Viewer for multi-slice, multi-b-value DWI DICOM files (uses the Matplotlib GUI
  widget)

Requirements
------------
- NumPy
- SciPy
- Scikit-Learn
- [leastsqbound-scipy](https://github.com/jjhelmus/leastsqbound-scipy) (if
  fitting)
- Matplotlib (if plotting)
- [Pydicom](https://code.google.com/p/pydicom/) (if handling DICOM files)

Todo or In Progress
-------------------
- Better documentation
- Proper pipelining
- Regression classification
- Autonomous tumor delineation/ROI placement
