"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [1, 2], [5, 0]], [5, 2]),
        ([[-1, 0], [2, -2], [4, -10]], [4, 0]),
        ([[-1, 0], [-5, -1], [0, -5]], [0, 0])
    ])
def test_daily_max(test, expected):
    """Test max function works for array of zeroes, positive and negative integers"""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 9], [0, 0], [5, 6]], [0, 0]),
        ([[-10, 0], [0, 0], [4, -2]], [-10, -2]),
        ([[-1, 0], [-5, -4], [0, -25]], [-5, -25])
    ])
def test_daily_min(test, expected):
    """Test that min function works for array of zeros, positive and negative integers"""
    from inflammation.models import daily_min
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                None,
        ),
        (
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                None,
        ),
        (
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
                None,
        ),
        (
                [[float('nan'), 0, 5], [float('nan'), float('nan'), float('nan')], [float('nan'), 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                None,
        ),
        (
                [[-5, -10, 1], [0, -20, 10], [-246, -3, 356]],
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                ValueError,
        )
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
           Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)