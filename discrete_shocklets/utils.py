import numpy as np


def diff(sequence, ghost=True):
    """
    Implements backwards difference, which is necessary to avoid looking forward in time

    :param sequence: sequence to difference
    :type sequence: iterable
    :param ghost:
    :type ghost: bool
    :returns: backwards difference given by :math:`x(n) - x(n-1)` of length `len(sequence) - 1`
    :rtype: numpy.ndarray

    """
    if not ghost:
        x = np.asarray(sequence)
        ret = np.asarray([x[n] - x[n - 1] for n in range(1, len(x))])
        return ret
    else:
        x = np.empty(len(sequence) + 1)
        x[1:] = sequence
        x[0] = sequence[0]
        ret = np.asarray([x[n] - x[n - 1] for n in range(1, len(x))])
        return ret


def make_seq_prediction_data(sequence, X_lag, y_lag):
    """
    Generates 2d train and test tensors for classification or regression

    :param sequence: 1d tensor that gives the sequence data
    :type sequence: numpy.ndarray
    :param X_lag: length of the X sequence (input) examples
    :type X_lag: int
    :param y_lag: length of the y sequence (to predict) examples
    :type y_lag: int
    :returns: X_lag, a (n - y_lag) x X_lag 2d tensor, and y_lag, a (n - X_lag) x y_lag 2d tensor
    :rtype: numpy.ndarray
    """
    X = make_moving_tensor(sequence[:-y_lag], X_lag)
    y = make_moving_tensor(sequence[X_lag:], y_lag)
    return X, y


def make_moving_tensor(tensor, lag):
    """
    Makes moving rank n+1 tensor from a rank n tensor.

    :param tensor: rank :math:`n` numpy ndarray
    :type tensor: numpy.ndarray
    :param lag: amount of time to be included; second dimension is :math:`T_{t},...,T_{t-lag}`
    :type lag: int
    :returns: a rank :math:`n+1` tensor
    :rtype: numpy.ndarray

    """
    lag = int(lag)
    return np.asarray([tensor[n: n + lag] for n in range(tensor.shape[0] - lag)])


def row_normalize(X):
    """ 
    Normalizes rows of a 2d tensor (matrix) to have unit variance and zero mean

    Defines a function :math:`\phi:\mathbb{R}^{n \\times m } \\rightarrow \mathbb{R}^{n \\times m}
    \\times \mathbb{R}^n \\times \mathbb{R}^n`
    This function together with ``row_unnormalize``, :math:`\psi`, give the identity:
    :math:`(\psi \circ \phi)(X) = X`. For example::

                In [1]: import numpy as np

                In [2]: import stat_tools as st

                In [3]: a = np.random.randn(4, 5)

                In [4]: b = st.row_unnormalize( *st.row_normalize(a) )

                In [5]: a == b
                Out[5]: 
                array([[ True,  True,  True,  True,  True],
                       [ True,  True,  True,  True,  True],
                       [ True,  True,  True,  True,  True],
                       [ True,  True,  True,  True,  True]], dtype=bool)

    :param X: rank 2 tensor
    :type X: numpy.ndarray
    :returns: (X, means, stds) normalized data, vector of mean values, vector of standard deviations
    :rtype: tuple

    """
    means = np.zeros(X.shape[0])
    stds = np.zeros(X.shape[0])

    for row in range(X.shape[0]):
        means[row] = np.mean(X[row])
        stds[row] = np.std(X[row])
        X[row] -= means[row]
        X[row] /= stds[row]

    return X, means, stds


def row_unnormalize(X, means, stds):
    """ 
    Takes a unit variance, zero mean rank 2 tensor with its mean and std vectors and unnormalizes it

    Defines a function :math:`\psi: \mathbb{R}^{n \\times m}
    \\times \mathbb{R}^n \\times \mathbb{R}^n \\rightarrow \mathbb{R}^{n \\times m }`
    This function together with ``row_normalize``, :math:`\phi`, give the identity:
    :math:`(\psi \circ \phi)(X) = X`.

    :param X: rank 2 tensor with unit variance and zero mean
    :type X: numpy.ndarray
    :param means: rank 1 tensor of the means taken across rows of X
    :type means: numpy.ndarray
    :param stds: rank 1 tensor of the stds taken across rows of X
    :type stds: numpy.ndarray
    :returns: un-normalized X
    :rtype: numpy.ndarray
    """
    for row in range(X.shape[0]):
        X[row] *= stds[row]
        X[row] += means[row]

    return X


def zero_norm(arr):
    """Normalizes an array so that it sums to zero

    Args:
      arr(iterable): the array

    Returns:
      : numpy.ndarray -- the zero-normalized array

    """
    arr = 2 * (arr - min(arr)) / (max(arr) - min(arr)) - 1
    return arr - np.sum(arr) / len(arr)


def normalize(arr, stats=False):
    """Normalizes an array to have zero mean and unit variance

    Args:
      arr(iterable): the array
      stats(bool, optional): if stats is True, will also return mean and std of array (Default value = False)

    Returns:
      : numpy.ndarray -- normalized array; or, if stats is True, tuple -- (normalized array, mean, std)

    """
    arr = np.array(arr)
    mean = arr.mean()
    std = arr.std()
    normed = (arr - mean) / std
    if not stats:
        return normed
    return normed, mean, std


def renormalize(arr, mean, std):
    arr = np.array(arr)
    return (arr - mean) / std


def top_k(indices, words, k):
    """Find the top k words by the weighted cusp indicator function

    Args:
      indices(list or numpy.array): a list of indicator values
      words(list or numpy.array): a list of strings
      k(int > 0): number of indices to look up.

    Returns:
      : list -- length k list of (word, indicator value) tuples

    """
    inds = np.argpartition(indices, -k)[-k:]
    topkwords = words[inds]
    topkvals = indices[inds]
    top = [(word, val) for word, val in zip(topkwords, topkvals)]
    top = sorted(top, key=lambda t: t[1], reverse=True)
    return top


def window_argmaxes(windows, data):
    """Find argmax point in data within each window.

    Args:
      windows(a 2D list or 2D numpy.ndarray): a list of indices indicating targeted windows in the data array
      data(ata: a list or numpy.ndarray): a list of data points
      data:

    Returns:
      : numpy.array -- max points for each window.

    """
    data = np.array(data)
    argmaxes = []

    for window in windows:
        data_segment = data[window]
        argmaxes.append(window[np.argmax(data_segment)])

    return np.array(argmaxes)