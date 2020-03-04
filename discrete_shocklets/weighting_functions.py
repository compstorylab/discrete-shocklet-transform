import numpy as np

registered_weighting_functions = []


def register_weighting_function(kernel_function):
    registered_weighting_functions.append(kernel_function)
    return kernel_function


@register_weighting_function
def max_change(arr):
    """Calculates the difference between the max and min points in an array.

    Args:
      arr(arr: a list or numpy.ndarray): a time series
      arr:

    Returns:
      : float -- maximum relative change

    """
    return np.max(arr) - np.min(arr)


@register_weighting_function
def max_rel_change(arr, neg=True):
    """Calculates the maximum relative changes in an array (log10).

    One possible choice for a weighting function in cusplet transform.

    Args:
      arr(arr: a list or numpy.ndarray): a time series for a given word
      neg: if true) arr - np.min(arr) + 1 (Default value = True)
      arr:

    Returns:
      : float -- maximum relative change (log10)

    """
    if neg:
        arr = arr - np.min(arr) + 1

    logret = np.diff(np.log10(arr))
    return np.max(logret) - np.min(logret)
