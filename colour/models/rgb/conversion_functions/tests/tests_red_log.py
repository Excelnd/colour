#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.conversion_functions.red_log`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.conversion_functions import (
    log_encoding_REDLog,
    log_decoding_REDLog,
    log_encoding_REDLogFilm,
    log_decoding_REDLogFilm)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_REDLog',
           'TestLogDecoding_REDLog',
           'TestLogDecoding_REDLogFilm',
           'TestLogDecoding_REDLogFilm']


class TestLogEncoding_REDLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLog` definition unit tests methods.
    """

    def test_log_encoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_REDLog(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_encoding_REDLog(0.18),
            0.63762184598817484,
            places=7)

        self.assertAlmostEqual(
            log_encoding_REDLog(1.0),
            1.0,
            places=7)

    def test_n_dimensional_log_encoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLog` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.63762184598817484
        np.testing.assert_almost_equal(
            log_encoding_REDLog(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_REDLog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_REDLog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_REDLog(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLog` definition nan support.
        """

        log_encoding_REDLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_REDLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLog` definition unit tests methods.
    """

    def test_log_decoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_REDLog(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_REDLog(0.63762184598817484),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_REDLog(1.0),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLog` definition n-dimensional arrays support.
        """

        V = 0.63762184598817484
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_REDLog(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_REDLog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_REDLog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_REDLog(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLog` definition nan support.
        """

        log_decoding_REDLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_REDLogFilm(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_encoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLogFilm` definition.
        """

        self.assertAlmostEqual(
            log_encoding_REDLogFilm(0.0),
            0.092864125122189639,
            places=7)

        self.assertAlmostEqual(
            log_encoding_REDLogFilm(0.18),
            0.45731961308541841,
            places=7)

        self.assertAlmostEqual(
            log_encoding_REDLogFilm(1.0),
            0.66959921798631472,
            places=7)

    def test_n_dimensional_log_encoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLogFilm` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.45731961308541841
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_encoding_REDLogFilm` definition nan support.
        """

        log_encoding_REDLogFilm(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_REDLogFilm(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_decoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLogFilm` definition.
        """

        self.assertAlmostEqual(
            log_decoding_REDLogFilm(0.092864125122189639),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_REDLogFilm(0.45731961308541841),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_REDLogFilm(0.66959921798631472),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLogFilm` definition n-dimensional arrays support.
        """

        V = 0.45731961308541841
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.red_log.\
log_decoding_REDLogFilm` definition nan support.
        """

        log_decoding_REDLogFilm(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
