#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rec. 2020 Colourspace
=====================

Defines the *Rec. 2020* colourspace:

-   :attr:`REC_2020_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Telecommunication Union. (2014). Parameter values for
        ultra-high definition television systems for production and
        international programme exchange. In Recommendation ITU-R BT.2020
        (Vol. 1, pp. 1–8). Retrieved from
        http://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2020-1-201406-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (
    RGB_Colourspace,
    normalised_primary_matrix,
    oecf_BT2020,
    eocf_BT2020)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['REC_2020_PRIMARIES',
           'REC_2020_ILLUMINANT',
           'REC_2020_WHITEPOINT',
           'REC_2020_TO_XYZ_MATRIX',
           'XYZ_TO_REC_2020_MATRIX',
           'REC_2020_COLOURSPACE']

REC_2020_PRIMARIES = np.array(
    [[0.708, 0.292],
     [0.170, 0.797],
     [0.131, 0.046]])
"""
*Rec. 2020* colourspace primaries.

REC_2020_PRIMARIES : ndarray, (3, 2)
"""

REC_2020_ILLUMINANT = 'D65'
"""
*Rec. 2020* colourspace whitepoint name as illuminant.

REC_2020_ILLUMINANT : unicode
"""

REC_2020_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(REC_2020_ILLUMINANT)
"""
*Rec. 2020* colourspace whitepoint.

REC_2020_WHITEPOINT : tuple
"""

REC_2020_TO_XYZ_MATRIX = normalised_primary_matrix(REC_2020_PRIMARIES,
                                                   REC_2020_WHITEPOINT)
"""
*Rec. 2020* colourspace to *CIE XYZ* tristimulus values matrix.

REC_2020_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_REC_2020_MATRIX = np.linalg.inv(REC_2020_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *Rec. 2020* colourspace matrix.

XYZ_TO_REC_2020_MATRIX : array_like, (3, 3)
"""

REC_2020_COLOURSPACE = RGB_Colourspace(
    'Rec. 2020',
    REC_2020_PRIMARIES,
    REC_2020_WHITEPOINT,
    REC_2020_ILLUMINANT,
    REC_2020_TO_XYZ_MATRIX,
    XYZ_TO_REC_2020_MATRIX,
    oecf_BT2020,
    eocf_BT2020)
"""
*Rec. 2020* colourspace.

REC_2020_COLOURSPACE : RGB_Colourspace
"""
