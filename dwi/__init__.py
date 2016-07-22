from __future__ import absolute_import, division, print_function


# Default runtime configuration parameters. Somewhat similar to matplotlib.
rcParamsDefault = {
        'texture.avg': False,  # Boolean: average result texture map?
        'texture.path': None,  # Write result directly to disk, if string.
        'texture.dtype': 'float32',  # Output texture map type.

        'texture.glcm.names': ('contrast', 'dissimilarity', 'homogeneity',
                               'energy', 'correlation', 'ASM'),
        'texture.glcm.distances': (1, 2, 3, 4),  # GLCM pixel distances.

        'texture.gabor.orientations': 6,  # Number of orientations.
        # 'texture.gabor.sigmas': (1, 2, 3),
        'texture.gabor.sigmas': (None,),
        'texture.gabor.freqs': (0.1, 0.2, 0.3, 0.4, 0.5),

        'texture.lbp.neighbours': 8,  # Number of neighbours.

        'texture.zernike.degree': 8,  # Maximum degree.

        'texture.haar.levels': 4,  # Numer of levels.
        }

# Modifiable runtime configuration parameters.
# TODO: Read from file.
rcParams = dict(rcParamsDefault)
