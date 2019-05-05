# -*- coding: utf-8 -*-
"""
Iridas .cube LUT Format Input / Output Utilities
================================================

Defines *Iridas* *.cube* *LUT* Format related input / output utilities objects.

-   :func:`colour.io.read_LUT_IridasCube`
-   :func:`colour.io.write_LUT_IridasCube`

References
----------
-   :cite:`AdobeSystems2013b` : Adobe Systems. (2013). Cube LUT
    Specification. Retrieved from https://drive.google.com/\
open?id=143Eh08ZYncCAMwJ1q4gWxVOqR_OSWYvs
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_INT_DTYPE
from colour.io.luts import LUT1D, LUT3x1D, LUT3D, LUTSequence
from colour.io.luts.common import parse_array, path_to_title
from colour.utilities import as_float_array, usage_warning

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['read_LUT_IridasCube', 'write_LUT_IridasCube']


def read_LUT_IridasCube(path):
    """
    Reads given *Iridas* *.cube* *LUT* file.

    Parameters
    ----------
    path : unicode
        *LUT* path.

    Returns
    -------
    LUT3x1D or LUT3d
        :class:`LUT3x1D` or :class:`LUT3D` class instance.

    References
    ----------
    :cite:`AdobeSystems2013b`

    Examples
    --------
    Reading a 3x1D *Iridas* *.cube* *LUT*:

    >>> import os
    >>> path = os.path.join(
    ...     os.path.dirname(__file__), 'tests', 'resources', 'iridas_cube',
    ...     'ACES_Proxy_10_to_ACES.cube')
    >>> print(read_LUT_IridasCube(path))
    LUT3x1D - ACES Proxy 10 to ACES
    -------------------------------
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (32, 3)

    Reading a 3D *Iridas* *.cube* *LUT*:

    >>> path = os.path.join(
    ...     os.path.dirname(__file__), 'tests', 'resources', 'iridas_cube',
    ...     'Colour_Correct.cube')
    >>> print(read_LUT_IridasCube(path))
    LUT3D - Generated by Foundry::LUT
    ---------------------------------
    <BLANKLINE>
    Dimensions : 3
    Domain     : [[ 0.  0.  0.]
                  [ 1.  1.  1.]]
    Size       : (4, 4, 4, 3)

    Reading a 3D *Iridas* *.cube* *LUT* with comments:

    >>> path = os.path.join(
    ...     os.path.dirname(__file__), 'tests', 'resources', 'iridas_cube',
    ...     'Demo.cube')
    >>> print(read_LUT_IridasCube(path))
    LUT3x1D - Demo
    --------------
    <BLANKLINE>
    Dimensions : 2
    Domain     : [[ 0.  0.  0.]
                  [ 1.  2.  3.]]
    Size       : (3, 3)
    Comment 01 : Comments can go anywhere
    """

    title = path_to_title(path)
    domain_min, domain_max = np.array([0, 0, 0]), np.array([1, 1, 1])
    dimensions = 3
    size = 2
    table = []
    comments = []

    with open(path) as cube_file:
        lines = cube_file.readlines()
        for line in lines:
            line = line.strip()

            if len(line) == 0:
                continue

            if line.startswith('#'):
                comments.append(line[1:].strip())
                continue

            tokens = line.split()
            if tokens[0] == 'TITLE':
                title = ' '.join(tokens[1:])[1:-1]
            elif tokens[0] == 'DOMAIN_MIN':
                domain_min = parse_array(tokens[1:])
            elif tokens[0] == 'DOMAIN_MAX':
                domain_max = parse_array(tokens[1:])
            elif tokens[0] == 'LUT_1D_SIZE':
                dimensions = 2
                size = DEFAULT_INT_DTYPE(tokens[1])
            elif tokens[0] == 'LUT_3D_SIZE':
                dimensions = 3
                size = DEFAULT_INT_DTYPE(tokens[1])
            else:
                table.append(parse_array(tokens))

    table = as_float_array(table)
    if dimensions == 2:
        return LUT3x1D(
            table,
            title,
            np.vstack([domain_min, domain_max]),
            comments=comments)
    elif dimensions == 3:
        # The lines of table data shall be in ascending index order,
        # with the first component index (Red) changing most rapidly,
        # and the last component index (Blue) changing least rapidly.
        table = table.reshape([size, size, size, 3], order='F')

        return LUT3D(
            table,
            title,
            np.vstack([domain_min, domain_max]),
            comments=comments)


def write_LUT_IridasCube(LUT, path, decimals=7):
    """
    Writes given *LUT* to given  *Iridas* *.cube* *LUT* file.

    Parameters
    ----------
    LUT : LUT3x1D or LUT3d or LUTSequence
        :class:`LUT3x1D`, :class:`LUT3D` or :class:`LUTSequence` class instance
        to write at given path.
    path : unicode
        *LUT* path.
    decimals : int, optional
        Formatting decimals.

    Returns
    -------
    bool
        Definition success.

    Warning
    -------
    -   If a :class:`LUTSequence` class instance is passed as ``LUT``, the
        first *LUT* in the *LUT* sequence will be used.

    References
    ----------
    :cite:`AdobeSystems2013b`

    Examples
    --------
    Writing a 3x1D *Iridas* *.cube* *LUT*:

    >>> from colour.algebra import spow
    >>> domain = np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]])
    >>> LUT = LUT3x1D(
    ...     spow(LUT3x1D.linear_table(16, domain), 1 / 2.2),
    ...     'My LUT',
    ...     domain,
    ...     comments=['A first comment.', 'A second comment.'])
    >>> write_LUT_IridasCube(LUT, 'My_LUT.cube')  # doctest: +SKIP

    Writing a 3D *Iridas* *.cube* *LUT*:

    >>> domain = np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]])
    >>> LUT = LUT3D(
    ...     spow(LUT3D.linear_table(16, domain), 1 / 2.2),
    ...     'My LUT',
    ...     np.array([[-0.1, -0.2, -0.4], [1.5, 3.0, 6.0]]),
    ...     comments=['A first comment.', 'A second comment.'])
    >>> write_LUT_IridasCube(LUT, 'My_LUT.cube')  # doctest: +SKIP
    """

    if isinstance(LUT, LUTSequence):
        LUT = LUT[0]
        usage_warning('"LUT" is a "LUTSequence" instance was passed, '
                      'using first sequence "LUT":\n'
                      '{0}'.format(LUT))

    assert not LUT.is_domain_explicit(), '"LUT" domain must be implicit!'

    if isinstance(LUT, LUT1D):
        LUT = LUT.as_LUT(LUT3x1D)

    assert (isinstance(LUT, LUT3x1D) or
            isinstance(LUT, LUT3D)), '"LUT" must be a 1D, 3x1D or 3D "LUT"!'

    is_3x1D = isinstance(LUT, LUT3x1D)

    size = LUT.size
    if is_3x1D:
        assert 2 <= size <= 65536, '"LUT" size must be in domain [2, 65536]!'
    else:
        assert 2 <= size <= 256, '"LUT" size must be in domain [2, 256]!'

    def _format_array(array):
        """
        Formats given array as an *Iridas* *.cube* data row.
        """

        return '{1:0.{0}f} {2:0.{0}f} {3:0.{0}f}'.format(decimals, *array)

    with open(path, 'w') as cube_file:
        cube_file.write('TITLE "{0}"\n'.format(LUT.name))

        if LUT.comments:
            for comment in LUT.comments:
                cube_file.write('# {0}\n'.format(comment))

        cube_file.write('{0} {1}\n'.format(
            'LUT_1D_SIZE' if is_3x1D else 'LUT_3D_SIZE', LUT.table.shape[0]))

        default_domain = np.array([[0, 0, 0], [1, 1, 1]])
        if not np.array_equal(LUT.domain, default_domain):
            cube_file.write('DOMAIN_MIN {0}\n'.format(
                _format_array(LUT.domain[0])))
            cube_file.write('DOMAIN_MAX {0}\n'.format(
                _format_array(LUT.domain[1])))

        if not is_3x1D:
            table = LUT.table.reshape([-1, 3], order='F')
        else:
            table = LUT.table

        for row in table:
            cube_file.write('{0}\n'.format(_format_array(row)))

    return True
