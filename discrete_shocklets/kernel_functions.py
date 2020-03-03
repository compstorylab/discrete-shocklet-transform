"""
API for kernel functions:
    `function(W, *args, zn=True)`
        W: Window size (in)
        *args: Additional positional arguments to the function as parameters (must be cast-able to float)
        zn: Boolean that indicates if the kernel function integrates to zero (default is True).
"""

import numpy as np

from discrete_shocklets.utils import zero_norm


def haar(L, zn=True):
    res = -1 * np.ones(L)
    res[len(res) // 2:] = 1
    if zn:
        return zero_norm(res)
    return res


def power_law_zero_cusp(L, b, zn=True, startpt=1, endpt=4):
    x = np.linspace(startpt, endpt, L)
    res = x ** (-b)
    res[:len(res) // 2] = 0
    if zn:
        return zero_norm(res)
    return res


def power_law_cusp(L, b, zn=True, startpt=1, endpt=4):
    res = power_law_zero_cusp(
        L,
        b,
        zn=zn,
        startpt=startpt,
        endpt=endpt
    ) + power_law_zero_cusp(
        L,
        b,
        zn=zn,
        startpt=startpt,
        endpt=endpt
    )[::-1]
    if zn:
        return zero_norm(res)
    return res


def power_cusp(L, b, zn=True, startpt=1, endpt=4):
    res = power_zero_cusp(
        L,
        b,
        zn=zn,
        startpt=startpt,
        endpt=endpt,
    ) + power_zero_cusp(
        L,
        b,
        zn=zn,
        startpt=startpt,
        endpt=endpt,
    )[::-1]
    if zn:
        return zero_norm(res)
    return res


def pitchfork(L, b, zn=True, startpt=1, endpt=4):
    res = power_zero_cusp(
        L,
        2 * b,
        zn=zn,
        startpt=startpt,
        endpt=endpt,
    )[::-1] + power_cusp(
        L,
        b,
        zn=zn,
        startpt=startpt,
        endpt=endpt
    ) + power_zero_cusp(
        L,
        2 * b,
        zn=zn,
        startpt=startpt,
        endpt=endpt,
    )
    if zn:
        return zero_norm(res)
    return res


def power_zero_cusp(L, b, zn=True, startpt=1, endpt=4):
    x = np.linspace(startpt, endpt, L)
    res = x ** b
    res[len(res) // 2:] = 0
    if zn:
        return zero_norm(res)
    return res


def exp_cusp(L, a, zn=True, startpt=1, endpt=4):
    res = exp_zero_cusp(
        L,
        a,
        zn=zn,
        startpt=startpt,
        endpt=endpt,
    ) + exp_zero_cusp(
        L,
        a,
        zn=zn,
        startpt=startpt,
        endpt=endpt,
    )[::-1]
    if zn:
        return zero_norm(res)
    return res


def exp_zero_cusp(L, a, zn=True, startpt=1, endpt=4):
    x = np.linspace(startpt, endpt, L)
    res = np.exp(a * x)
    res[len(res) // 2:] = 0
    if zn:
        return zero_norm(res)
    return res
