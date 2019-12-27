# -*- coding: utf-8 -*-
"""
Colour Gamut Plotting
=====================

Defines the colour gamut plotting objects:

-   :func:`plot_gamut_boundary_descriptors_hue_segments`
"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

from colour.constants import DEFAULT_INT_DTYPE
from colour.gamut import sample_gamut_boundary_descriptor, spherical_to_Jab
from colour.models import Jab_to_JCh
from colour.plotting import COLOUR_STYLE_CONSTANTS, override_style, render
from colour.utilities import as_float_array, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['plot_gamut_boundary_descriptors_hue_segments']


@override_style()
def plot_gamut_boundary_descriptors_hue_segments(GBD_m,
                                                 hue_angles=None,
                                                 columns=None,
                                                 **kwargs):
    GBD_m = [as_float_array(GBD_m_c) for GBD_m_c in GBD_m]

    assert len(np.unique([GBD_m_c.shape for GBD_m_c in GBD_m])) <= 3, (
        'Gamut boundary descriptor matrices have incompatible shapes!')

    settings = {}
    settings.update(kwargs)

    if hue_angles is not None:
        x_s = np.linspace(0, 180, GBD_m[0].shape[0])
        y_s = hue_angles
        x_s_g, y_s_g = np.meshgrid(x_s, y_s, indexing='ij')
        theta_alpha = tstack([x_s_g, y_s_g])

        GBD_m = [
            sample_gamut_boundary_descriptor(GBD_m_c, theta_alpha)
            for GBD_m_c in GBD_m
        ]

    GBD_m = [Jab_to_JCh(spherical_to_Jab(GBD_m_c)) for GBD_m_c in GBD_m]

    shape_r, shape_c, = GBD_m[0].shape[0], GBD_m[0].shape[1]

    columns = (shape_c
               if columns is None else max(len(hue_angles or []), columns))

    figure, axes_a = plt.subplots(
        DEFAULT_INT_DTYPE(np.ceil(shape_c / columns)),
        columns,
        sharex='all',
        sharey='all',
        gridspec_kw={
            'hspace': 0,
            'wspace': 0
        },
        constrained_layout=True,
    )

    axes_a = np.ravel(axes_a)

    for i in range(shape_c):
        label = '{0:d} $^\\degree$'.format(
            DEFAULT_INT_DTYPE((
                i / shape_c * 360) if hue_angles is None else hue_angles[i]))

        axes_a[i].text(
            0.5,
            0.5,
            label,
            alpha=COLOUR_STYLE_CONSTANTS.opacity.low,
            fontsize='xx-large',
            horizontalalignment='center',
            verticalalignment='center',
            transform=axes_a[i].transAxes)

        if i % columns == 0:
            axes_a[i].set_ylabel('J')

        if i > shape_c - columns:
            axes_a[i].set_xlabel('C')

        for j in range(len(GBD_m)):
            axes_a[i].plot(
                GBD_m[j][..., i, 1],
                GBD_m[j][..., i, 0],
                label='GBD {0}'.format(j))

        if i == shape_c - 1:
            axes_a[i].legend()

    for axes in axes_a[shape_c:]:
        axes.set_visible(False)

    settings = {
        'figure_title':
            'Gamut Boundary Descriptors - {0} Hue Segments'.format(shape_c),
        'tight_layout':
            False,
    }
    settings.update(kwargs)

    render(**settings)
