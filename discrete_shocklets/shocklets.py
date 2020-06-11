import numpy as np
from scipy import signal

from . import kernel_functions
from . import utils


def cusplet(
        arr,
        widths,
        kernel_args=None,
        kernel_func=kernel_functions.power_cusp,
        method='fft',
        reflection=0,
        width_weights=None,
):
    """Implements the discrete cusplet transform.

    Args:
        arr(list): array of shape (n,) or (n,1).
            This array should not contain inf-like values.
            The transform will still be computed but infs propagate.
            Nan-like values will be linearly interpolated, which is okay for subsequent
            time-based analysis but will introduce ringing in frequency-based analyses.
        widths(iterable): iterable of integers that specify the window widths (L above).
            Assumed to be in increasing order.
            If widths is not in increasing order the results will be garbage.
        kernel_args(list or tuple, optional): arguments for the kernel function.
        kernel_func(callable): A kernel factory function.
            See kernel_functions.py for the required interface and available options.
        method(str, optional): one of 'direct' or 'fft' (Default value = 'fft')
        reflection(int, optional): Element of the reflection group applied to the kernel function.
            Default is 0, corresponding to the identity element.
        width_weights(list or None, optional): Relative importance of the different window widths.

    Returns:
      : tuple -- (numpy array of shape (L, n) -- the cusplet transform, k -- the calculated kernel function)

    """
    if kernel_args is None:
        kernel_args = []
    elif type(kernel_args) is float:
        kernel_args = [kernel_args]
    if width_weights is None:
        width_weights = np.ones_like(widths)
    else:
        width_weights = np.array(width_weights)

    arr = utils.fill_na(np.array(arr), mode='interpolate')
    cc = np.zeros((len(widths), len(arr)))

    for i, width in enumerate(widths):
        kernel = kernel_func(width, *kernel_args)
        kernel = utils.apply_reflection_action(kernel, reflection)
        cc[i] = signal.correlate(arr, kernel, mode='same', method=method)
    cc = width_weights[..., np.newaxis] * cc
    return cc, kernel


def cusplet_parameter_sweep(
        arr,
        widths,
        kernel_weights=None,
        kernel_args=None,
        kernel_func=kernel_functions.power_cusp,
        reflection=0,
        width_weights=None,
):
    """Sweeps over values of parameters (kernel arguments) in the discrete cusplet transform.

    Args:
      arr(list): numpy array of shape (n,) or (n,1), time series
      kernel_func(callable): kernel function. Must take an integer L > 0 as an argument and any number of additional, nonkeyword arguments, and returns a numpy array of shape (L,) that implements the kernel. The returned array should sum to zero; use the zero_norm function for this.
      widths(iterable): iterable of integers that specify the window widths (L above). Assumed to be in increasing order; if widths is not in increasing order the results will be garbage.
      kernel_args(list or tuple of lists or tuples): iterable of iterables of arguments for the kernel function. Each top-level iterable is treated as a single parameter vector.
      reflection(int, optional): integer n evaluates to n %4, element of the reflection group that left-multiplies the kernel function. Default is 0 (identity element).
      width_weights (list or None, optional):
      kernel_weights(list or None, optional):

    Returns:
      : numpy.ndarray -- numpy array of shape (L, n, len(k_args)), the cusplet transform

    """
    kernel_args = np.array(kernel_args)

    if kernel_weights is None:
        kernel_weights = np.ones(kernel_args.shape[0])

    cc = np.zeros((len(widths), len(arr), len(kernel_args)))

    for i, k_arg in enumerate(kernel_args):
        cres, _ = cusplet(
            arr,
            widths,
            kernel_args=k_arg,
            kernel_func=kernel_func,
            reflection=reflection,
            width_weights=width_weights,
        )
        cc[:, :, i] = cres * kernel_weights[i]

    return cc


def classify_cusps(cc, b=1, geval=False):
    """Classifies points as belonging to cusps or not.

    Args:
      cc(numpy.ndarray): numpy array of shape (L, n), the cusplet transform of a time series
      b(int or float, optional): multiplier of the standard deviation. (Default value = 1)
      geval(float >= 0, optional): If geval is an int or float, classify_cusps will return (in addition to the cusps and cusp intensity function) an array of points where the cusp intensity function is greater than geval. (Default value = False)

    Returns:
      : tuple --- (numpy.ndarray of indices of the cusps; numpy.ndarray representing the cusp intensity function) or, if geval is not False, (extrema; the cusp intensity function; array of points where the cusp intensity function is greater than geval)

    """
    sum_cc = utils.zero_norm(np.nansum(cc, axis=0))
    mu_cc = np.nanmean(sum_cc)
    std_cc = np.nanstd(sum_cc)

    extrema = np.array(signal.argrelextrema(sum_cc, np.greater))[0]
    extrema = [x for x in extrema if sum_cc[x] > mu_cc + b * std_cc]

    if geval is False:
        return extrema, sum_cc
    else:
        gez = np.where(sum_cc > geval)
        return extrema, sum_cc, gez


def _make_components(indicator, cusp_points=None):
    """Get individual windows from array of indicator indices.
    
    Takes cusp indicator function and returns windows of contiguous cusp-like behavior.
    If an array of hypothesized deterministic peaks of cusp-like behavior is passed,
    thins these points so that there is at most one point per window.

    Args:
      indicator(list): array of the points where the cusp intensity function exceeds some threshold
      cusp_points(list or numpy.ndarray, optional): optional, array of points that denote the hypothesized deterministic peaks of cusps (Default value = None)

    Returns:
      list -- the contiguous cusp windows; or, if cusp_points is not None, tuple -- (the contiguous cusp windows, the thinned cusp points)

    """
    windows = []
    indicator = np.array(indicator)
    if len(indicator.shape) > 1:
        indicator = indicator[0]
    j = 0

    for i, x in enumerate(indicator):
        if i == len(indicator) - 1:
            window = indicator[j: i]
            if len(window) >= 2:
                windows.append(window)
            break
        elif indicator[i + 1] == x + 1:
            continue  # still part of the same block
        else:  # block has ended
            window = indicator[j: i]
            if len(window) >= 2:
                windows.append(window)
            j = i + 1

    if cusp_points is None:
        return windows

    pt_holder = [[] for _ in range(len(windows))]
    for pt in cusp_points:
        for i, window in enumerate(windows):
            if (pt >= window[0]) and (pt <= window[-1]):
                pt_holder[i].append(pt)
                break

    windows_ = []
    estimated_cusp_points = []
    for holder, window in zip(pt_holder, windows):
        if holder:
            windows_.append(window)
            estimated_cusp_points.append(int(np.median(holder)))

    estimated_cusp_points = np.array(estimated_cusp_points, dtype=int)

    return windows_, estimated_cusp_points


def make_components(indicator, cusp_points=None, scan_back=0):
    """Get individual windows from array of indicator indices.
    
    Takes cusp indicator function and returns windows of contiguous cusp-like behavior.
    If an array of hypothesized deterministic peaks of cusp-like behavior is passed,
    thins these points so that there is at most one point per window.
    The scan_back parameter connects contiguous windows if they are less than or equal to
    scan_back indices from each other.

    Args:
      indicator(list): array of the points where the cusp intensity function exceeds some threshold
      cusp_points(list or numpy.ndarray, optional): optional, array of points that denote the hypothesized deterministic peaks of cusps (Default value = None)
      scan_back(int >= 0, optional): number of indices to look back. If cusp windows are within scan_back indices of each other, they will be connected into one contiguous window. (Default value = 0)

    Returns:
      list -- the contiguous cusp windows; or, if cusp_points is not None, tuple -- (the contiguous cusp windows, the thinned cusp points)

    """
    windows = _make_components(indicator, cusp_points=cusp_points)

    if cusp_points is not None:
        windows, estimated_cusp_points = windows

    if (len(windows) > 1) and (scan_back > 0):
        windows_ = []
        for i in range(len(windows)):
            if len(windows_) == 0:
                windows_.append(list(windows[i]))
            else:
                if windows[i][0] <= windows_[-1][-1] + scan_back:
                    fill_between = list(range(windows_[-1][-1] + 1,
                                              windows[i][0]))
                    windows_[-1].extend(fill_between)
                    windows_[-1].extend(list(windows[i]))
                else:
                    windows_.append(list(windows[i]))

    else:
        windows_ = windows

    if cusp_points is None:
        return windows_
    return windows_, estimated_cusp_points


def setup_corr_mat(k, N):
    """Sets up linear operator corresponding to cross correlation.
    
    The cross-correlation operation can be just viewed as a linear operation from R^K to R^K as
    Ax = C. The operator A is a banded matrix that represents the rolling add operation that
    defines cross-correlation. To execute correlation of the kernel with data
    one computes np.dot(A, data).

    Args:
      k(numpy.ndarray): the cross-correlation kernel
      N(positive int): shape of data array with which k will be cross-correlated.

    Returns:
      numpy.ndarray -- NxN array, the cross-correlation operator

    """

    def _sliding_windows(a, N):
        """Generates band numpy array *quickly*
        Taken from https://stackoverflow.com/questions/52463972/generating-banded-matrices-using-numpy.

        Args:
          a:
          N:

        Returns:

        """
        a = np.asarray(a)
        p = np.zeros(N - 1, dtype=a.dtype)
        b = np.concatenate((p, a, p))
        s = b.strides[0]
        return np.lib.stride_tricks.as_strided(
            b[N - 1:],
            shape=(N, len(a) + N - 1),
            strides=(-s, s),
        )

    full_corr_mat = _sliding_windows(k, N)
    overhang = full_corr_mat.shape[-1] - N
    if overhang % 2 == 1:
        front = int((overhang + 1) / 2) - 1
        back = front + 1
    else:
        front = back = int(overhang / 2)
    corr_mat = full_corr_mat[:, front:-back]

    return corr_mat


def matrix_cusplet(
        arr,
        widths,
        kernel_func=kernel_functions.power_cusp,
        kernel_args=None,
        reflection=0,
        width_weights=None,
):
    """Computes the cusplet transform using matrix multiplication.
    
    This method is provided for the sake only of completeness; it is orders of magnitude
    slower than ``cusplets.cusplet`` and there is no good reason to use it in production.
    You should use ``cusplets.cusplet`` instead.

    Args:
      arr(list, tuple, or numpy.ndarray): array of shape (n,) or (n,1). This array should not contain inf-like values. The transform will still be computed but infs propagate. Nan-like values will be linearly interpolated, which is okay for subsequent time-based analysis but will introduce ringing in frequency-based analyses.
      widths(iterable): iterable of integers that specify the window widths (L above). Assumed to be in increasing order; if widths is not in increasing order the results will be garbage.
      kernel_func(callable): kernel function. Must take an integer L > 0 as an argument and any number of additional, nonkeyword arguments, and returns a numpy array of shape (L,) that implements the kernel. The returned array should sum to zero; use the zero_norm function for this.
      kernel_args(list or tuple, optional): arguments for the kernel function. (Default value = None)
      reflection(int, optional): integer n evaluates to n %4, element of the reflection group that left-multiplies the kernel function. Default is 0 (identity element).
      width_weights: type width_weights: (Default value = None)

    Returns:
      tuple -- (numpy array of shape (L, n) -- the cusplet transform, None)

    """
    if kernel_args is None:
        kernel_args = []
    if width_weights is None:
        width_weights = np.ones_like(widths)
    else:
        width_weights = np.array(width_weights)

    arr = utils.fill_na(np.array(arr))
    cc = np.zeros((len(widths), len(arr)))

    for i, width in enumerate(widths):
        kernel = kernel_func(width, *kernel_args)
        kernel = utils.apply_reflection_action(kernel, reflection)

        # Set up the cross correlation
        corr_mat = setup_corr_mat(kernel, arr.shape[0])
        cc[i] = np.dot(corr_mat, arr)
    cc = width_weights[..., np.newaxis] * cc
    return cc, kernel


def inverse_cusplet(
        cc,
        kernel,
        widths,
        k_args=None,
        reflection=0,
        width_ind=0,
):
    """Computes the inverse of the discrete cusplet / shocklet transform.
    
    The cusplet transform is overcomplete, at least in theory. Since each row of the cusplet transform is
    a cross-correlation between the kernel function and the time series, it is---again, in theory---possible to recover
    the original time series of data from any row of the cusplet transform and the appropriate kernel.
    A row of the cusplet transform, denoted by c, is defined by c = Ax, where A is the cross-correlation matrix
    constructed from kernel k.
    If A is invertible then we trivially have x = A^{-1}c.
    In theory, the full inverse transform using the full cusplet transform C is given by
    x = \langle A_w^{-1}c_w\\rangle_{w}, where by w we denote the appropriate kernel width parameter.
    
    We note that there is really no reason to ever use this function. Unlike other transforms,
    it is *highly* unlikely that a user will be confronted with some arbitrary cusplet transform and need to
    recover the raw data from it.
    In other words, one often is confronted with frequency data corresponding to a Fourier transform and needs to
    extract the original time series from it, but the cusplet transform is intended to be a data analysis tool and
    so the data should always be accessible to the user.
    
    In the practical implementation here,
    the user can specify which row of the cusplet transform to use. By default we will use the first row of the
    transform since this will introduce the fewest numerical errors in the inversion.
    This is true because the convolution operations involves fewer elements in each sum;
    the convolution matrix will have lower bandwidth and hence will be easier to invert.

    Args:
      cc(numpy.ndarray): the cusplet transform array, shape W x T
      kernel(callable): kernel function. Must take an integer L > 0 as an argument and any number of additional, nonkeyword arguments, and returns a numpy array of shape (L,) that implements the kernel. The returned array should sum to zero; use the zero_norm function for this.
      widths(iterable): iterable of integers that specify the window widths (L above). Assumed to be in increasing order; if widths is not in increasing order the results will be garbage.
      k_args(list or tuple, optional): arguments for the kernel function. (Default value = None)
      reflection(int, optional): integer n evaluates to n %4, element of the reflection group that left-multiplies the kernel function. Default is 0 (identity element).
      width_ind: type width_ind: (Default value = 0)

    Returns:
      numpy.ndarray -- the reconstructed original time series. This time series will have (roughly) the same functional form as the original, but it is not guaranteed that its location and scale will be the same.

    """
    if k_args is None:
        k_args = []

    # now we will see what group action to operate with
    # cusplet transform is overcomplete so we need only invert one row
    # by default choose the one with smallest kernel as will have
    # smallest numerical error
    w = widths[width_ind]
    k = kernel(w, *k_args)
    # implement reflections
    reflection = reflection % 4
    if reflection == 1:
        k = k[::-1]
    elif reflection == 2:
        k = -k
    elif reflection == 3:
        k = -k[::-1]
    corr_mat = setup_corr_mat(k, cc.shape[1])
    # cusplet transform can be written Ax = C
    # so we need x = A^{-1}C
    # but this is gross and expensive so solve using lstsq
    invq = np.linalg.lstsq(
        corr_mat,
        cc[width_ind],
        rcond=-1
    )
    return invq
