# -*- coding: utf-8 -*-
"""
Colour Gamut Plotting
=====================

Defines the colour gamut plotting objects:

-   :func:`colour.plotting.plot_segment_maxima_gamut_boundary_segments`
-   :func:`colour.plotting.plot_segment_maxima_gamut_boundary`
-   :func:`colour.plotting.plot_Jab_colours_in_segment_maxima_gamut_boundary`
"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from colour.algebra import polar_to_cartesian
from colour.constants import DEFAULT_INT_DTYPE
from colour.gamut import (gamut_boundary_descriptor,
                          sample_gamut_boundary_descriptor, spherical_to_Jab)
from colour.models import Jab_to_JCh
from colour.plotting import (COLOUR_STYLE_CONSTANTS, artist, override_style,
                             render)
from colour.utilities import as_float_array, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'plot_segment_maxima_gamut_boundary_segments',
    'plot_segment_maxima_gamut_boundary',
    'plot_Jab_colours_in_segment_maxima_gamut_boundary'
]


@override_style()
def plot_segment_maxima_gamut_boundary_segments(GBD_m,
                                                columns=None,
                                                angles=None,
                                                **kwargs):
    GBD_m = [as_float_array(GBD_m_c) for GBD_m_c in GBD_m]

    assert len(np.unique([GBD_m_c.shape for GBD_m_c in GBD_m])) <= 3, (
        'Gamut boundary descriptor matrices have incompatible shapes!')

    settings = {}
    settings.update(kwargs)

    if angles is not None:
        x_s = np.linspace(0, 180, GBD_m[0].shape[0])
        y_s = angles
        x_s_g, y_s_g = np.meshgrid(x_s, y_s, indexing='ij')
        theta_alpha = tstack([x_s_g, y_s_g])

        GBD_m = [
            sample_gamut_boundary_descriptor(GBD_m_c, theta_alpha)
            for GBD_m_c in GBD_m
        ]

    GBD_m = [Jab_to_JCh(spherical_to_Jab(GBD_m_c)) for GBD_m_c in GBD_m]

    shape_r, shape_c, = GBD_m[0].shape[0], GBD_m[0].shape[1]

    columns = (shape_c if columns is None else max(len(angles or []), columns))

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
                i / shape_c * 360) if angles is None else angles[i]))

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

    return render(**settings)


@override_style()
def plot_segment_maxima_gamut_boundary(GBD_m,
                                       plane='Ja',
                                       gamut_boundary_descriptor_kwargs=None,
                                       **kwargs):
    GBD_m = as_float_array(GBD_m)
    plane = plane.lower()

    settings = {'uniform': True}
    settings.update(kwargs)

    gamut_boundary_descriptor_settings = {'E': np.array([50, 0, 0])}
    gamut_boundary_descriptor_settings.update(gamut_boundary_descriptor_kwargs)
    gamut_boundary_descriptor_settings.update({
        'close_callable': None,
        'fill_callable': None
    })

    figure, axes = artist(**settings)

    Jab = spherical_to_Jab(GBD_m)
    Jab += gamut_boundary_descriptor_settings['E']

    if plane.lower() == 'ab':
        x_label, y_label = 'a', 'b'

        J_p, a_p, b_p = tsplit(Jab[np.argmax(GBD_m[:, :, 0], 0),
                                   np.arange(GBD_m.shape[1])])
        x, y = a_p, b_p

        axes.plot(
            np.hstack([x, x[0]]),
            np.hstack([y, y[0]]),
            'o-',
            color=COLOUR_STYLE_CONSTANTS.colour.dark)

        segments = GBD_m.shape[1] + 1
    else:
        x_label, y_label = 'a' if plane == 'ja' else 'b', 'J'

        Jab[..., 2 if plane == 'ja' else 1] = 0
        settings = kwargs.copy()
        settings.update(kwargs)

        GBD_m_p = gamut_boundary_descriptor(
            Jab,
            m=GBD_m.shape[0],
            n=GBD_m.shape[1],
            **gamut_boundary_descriptor_settings).reshape(-1, 3)
        GBD_m_p = spherical_to_Jab(GBD_m_p[np.any(~np.isnan(GBD_m_p),
                                                  axis=-1)])
        GBD_m_p = np.vstack([GBD_m_p[0::2], GBD_m_p[1::2][::-1]])

        J_p, a_p, b_p = tsplit(GBD_m_p)
        x, y = (a_p if plane == 'ja' else b_p), J_p

        axes.plot(
            np.hstack([x, x[0]]),
            np.hstack([y, y[0]]),
            'o-',
            color=COLOUR_STYLE_CONSTANTS.colour.dark)

        segments = GBD_m.shape[0] * 2 + 1

    if kwargs.get('show_debug_circles'):
        for i in range(len(x)):
            axes.add_artist(
                plt.Circle(
                    [0, 0],
                    np.linalg.norm([x[i], y[i]]),
                    fill=False,
                    alpha=COLOUR_STYLE_CONSTANTS.opacity.low))

    rho = np.ones(segments) * 1000
    phi = np.radians(np.linspace(-180, 180, segments))

    lines = LineCollection(
        [((0, 0), a) for a in polar_to_cartesian(tstack([rho, phi]))],
        colors=COLOUR_STYLE_CONSTANTS.colour.dark,
        alpha=COLOUR_STYLE_CONSTANTS.opacity.low,
    )
    axes.add_collection(lines, )

    settings = {
        'title': 'Gamut Boundary Descriptor',
        'x_label': x_label,
        'y_label': y_label,
    }
    settings.update(kwargs)

    return render(**settings)


@override_style()
def plot_Jab_colours_in_segment_maxima_gamut_boundary(
        Jab,
        plane='Ja',
        gamut_boundary_descriptor_kwargs=None,
        scatter_kwargs=None,
        **kwargs):
    Jab = as_float_array(Jab)
    plane = plane.lower()
    if gamut_boundary_descriptor_kwargs is None:
        gamut_boundary_descriptor_kwargs = {}

    settings = {'uniform': True}
    settings.update(kwargs)

    figure, axes = artist(**settings)

    gamut_boundary_descriptor_settings = {'E': np.array([50, 0, 0])}
    gamut_boundary_descriptor_settings.update(gamut_boundary_descriptor_kwargs)

    GBD_m = gamut_boundary_descriptor(Jab,
                                      **gamut_boundary_descriptor_settings)

    settings.update({
        'axes': axes,
        'standalone': False,
    })

    figure, axes = plot_segment_maxima_gamut_boundary(
        GBD_m,
        plane=plane,
        gamut_boundary_descriptor_kwargs=gamut_boundary_descriptor_kwargs,
        **settings)

    axes.autoscale(False)

    J, a, b = tsplit(Jab - gamut_boundary_descriptor_settings['E'])

    if plane == 'ab':
        x, y = a, b
    else:
        x, y = (a, J) if plane == 'ja' else (b, J)

    scatter_settings = {
        # 's': 40,
        # 'c': 'RGB',
        's': 20,
        'marker': 'o',
        'alpha': 0.85,
    }
    if scatter_kwargs is not None:
        scatter_settings.update(scatter_kwargs)

    axes.scatter(x, y, **scatter_settings)

    settings.update({'standalone': True})
    settings.update(kwargs)

    return render(**settings)
