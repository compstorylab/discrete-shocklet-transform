import numpy as np
import pytest

from discrete_shocklets import kernel_functions, shocklets


def test_cusplet_kernel_args():
    arr = np.random.randn(100)
    kernel = kernel_functions.power_zero_cusp
    widths = [50]
    # need kernel args but we don't give them
    with pytest.raises(TypeError) as e_info:
        _ = shocklets.cusplet(arr, widths, kernel_func=kernel)


def test_cusplet_output():
    arr = np.random.randn(100)
    kernel = kernel_functions.power_zero_cusp
    widths = np.linspace(10, 30, 20).astype(int)
    k_args = [3.]
    cc, _ = shocklets.cusplet(arr, widths, kernel_func=kernel, kernel_args=k_args)
    assert cc.shape == (20, 100)


def test_classify_cusps():
    arr1 = np.linspace(1, 100, 100)
    arr2 = np.linspace(100, 1, 100)
    arr = np.hstack((arr1, arr2))  # make a triangle
    kernel = kernel_functions.power_cusp
    widths = np.linspace(10, 30, 20).astype(int)
    k_args = [3.]
    cc, _ = shocklets.cusplet(arr, widths, kernel_func=kernel, kernel_args=k_args)
    extrema, sum_cc, gez = shocklets.classify_cusps(cc, b=0.75, geval=0.5)
    assert len(gez) > 0
    assert max(sum_cc) > 1
    assert extrema[0] == 100


def test_make_components():
    arr1 = np.linspace(1, 100, 100)
    arr2 = np.linspace(100, 1, 100)
    arr = np.hstack((arr1, arr2))  # make a triangle
    kernel = kernel_functions.power_cusp
    widths = np.linspace(10, 30, 20).astype(int)
    k_args = [3.]
    cc, _ = shocklets.cusplet(arr, widths, kernel_func=kernel, kernel_args=k_args)
    extrema, sum_cc, gez = shocklets.classify_cusps(cc, b=0.75, geval=0.5)
    windows, estimated_cusp_pts = shocklets.make_components(gez, extrema)
    assert len(windows) == 1
