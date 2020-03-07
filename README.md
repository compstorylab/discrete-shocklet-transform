
# Discrete Shocklet Transform (DST) and Shocklet Transform And Ranking (STAR) algorithm
## A qualitative, shape-based, timescale-invariant methodology for finding shapes in sociotechnical data


Without further ado (for further ado, see below), here is the help string for the main code, the STAR algorithm:

```
star -h
usage: star [-h] [-i INPUT] [-e ENDING] [-d DELIMITER] [-o OUTPUT]
            [-k KERNEL] [-r REFLECTION] [-b BVALUE] [-g GEVAL]
            [-l LOOKBACK] [-w WEIGHTING] [-wmin WMIN] [-wmax WMAX] [-nw NW]
            [-s SAVESPEC] [-norm NORM]

Runs the Shocklet Transform And Ranking (STAR) algorithm on data

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to files on which to run the algorithm. 
                                    The files should be in row-major order. That is, 
                                    they should be a N_variable x T matrix, where T is 
                                    the number of timesteps. This algorithm multiprocesses over 
                                    files so if your time series are very long it could be significantly
                                    faster to break them up into multiple files, one time series per file.
                                    (This is true only if you have many cores on which processing can occur 
                                    in parallel.)
  -e ENDING, --ending ENDING
                        Ending of files on which to run algorithm. Must be readable to numpy.genfromtxt()
  -d DELIMITER, --delimiter DELIMITER
                        Delimiter of entries in files
  -o OUTPUT, --output OUTPUT
                        Path to which to save output
  -k KERNEL, --kernel KERNEL
                        Kernel function to use. Must be in dir(cusplets). Pre-written options (no 
                                    need to write code) are: haar, which is the Haar wavelet and looks for pure level 
                                    changes; power_zero_cusp, which is the building block for other cusp kernels and 
                                    looks for power (monomial) growth followed by an abrupt drop to constant low levels; 
                                    power_cusp, which is a power cusp shape; exp_zero_cusp, which is like 
                                    power_zero_cusp except with exponential growth; and exp_cusp, which is like 
                                    power_cusp except with exponential growth. Combining these kernels with a reflection 
                                    (computed using the option `-r <int>`) is probably enough to look for any interesting 
                                    behavior. If you want to write your own kernel function, it must be in cusplets.py and 
                                    conform to the API `function(W, *args, zn=True)` where W is a window size, 
                                    *args are any remaining positional arguments to the function as parameters and must 
                                    be cast-able to `float`, and zn is a boolean corresponding to whether or not to 
                                    ensure that the kernel function integrates to zero, which it should by default. 
                                    Kernel defaults to power_cusp.
  -r REFLECTION, --reflection REFLECTION
                        Element of the reflection group R_4 to use. Default is 0 (id). Computed mod 4.
  -b BVALUE, --bvalue BVALUE
                        Multiplier for std dev in classification. Default is b=0.75.
  -g GEVAL, --geval GEVAL
                        Threshold for window construction. Default is 0.5.
  -l LOOKBACK, --lookback LOOKBACK
                        Number of indices to look back for window construction. Default is 0.
  -w WEIGHTING, --weighting WEIGHTING
                        Method for weighting of cusp indicator functions. Must be in dir(cusplets). 
                                    Defaults to max_change, the maximum minus the minimum value of original series 
                                    within each window. The other pre-written option (no need to write code) is 
                                    max_rel_change, which computes max_change on the array of log returns of the 
                                    original time series. If you want to write your own weighting function, it must 
                                    be in cusplets.py and correspond to the API `function(arr)` where arr is an array.
  -wmin WMIN, --wmin WMIN
                        Smallest kernel size. Defaults to 10.
  -wmax WMAX, --wmax WMAX
                        Largest kernel size. Defaults to min{500, 1/2 length of time series}.
  -nw NW, --nw NW       Number of kernels to use. Ideally (wmax - wmin) / nw would be an integer. Default is 100.
  -s SAVESPEC, --savespec SAVESPEC
                        Spec for saving. Options are: cc, to just save cusplet transform; 
                                    indic, to save indicator function; windows, to save anomalous windows; 
                                    weighted, to save weighted indicator function; 
                                    all, to save everything. Defaults to all. Files are saved in compressed 
                                    .npz numpy archive format.
  -norm NORM, --norm NORM
                        Whether or not to normalize series to be wide-sense stationary 
                                    with intertemporal zero mean and unit variance. Default is False.
```

A key point: *any additional arguments passed to `star` after the named arguments above are interpreted as arguments to be passed to the kernel function*. So, if you're using the default settings and run something like `star -i ./_test/ -e csv`, you'll get an error that says

```
Error occurred in computation of shocklet transform of test
Error: power_cusp() missing 1 required positional argument: 'b'
```

This is expected behavior: you have to pass this argument in at the end, so instead you'd enter `star -i ./_test/ -e csv 3` and everything would be fine.

# More technical details

Further ado (_viz_: what is actually happening under the hood) proceeds below.

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import cusplets
```

# Discrete Shocklet Transform

This provides a few examples of the discrete shocklet (affectiontely referred to as a cusplet by some...) transform---or DST---in practice. We will show how the group action of $`R_4`$ on the kernel function affects the output of the transform and demonstrate the automated extraction of anomalous dynamics using the post-processing algorithms. You can find theoretical information about this collection of algorithms---and some applications---in the original paper: http://compstorylab.org/share/papers/dewhurst2019a/. 

First we will show what a cusplet transform actually looks like. We will use a pretty boring time series for this: just a random walk, $`x_n - x_{n-1} = u_n`$, where $`u_n \sim \mathcal{U}(\{-1, 1\})`$.


```python
np.random.seed( 50 )
noise = np.random.choice([-1, 1], size=5000, replace=True)
x = np.cumsum( noise )
plt.plot(x, 'darkgray')
plt.xlabel('$n$')
plt.ylabel('$x_n$')
plt.xlim(0, 5000 - 1);
```


![png](./example/output_2_0.png)


At its core, the DST is just simple cross-correlation of a kernel function $`\mathcal{K}`$ with a signal such as $`x_n`$. A windowing parameter $W$ controls up- and down-sampling time so that anomalous dynamics can be found at all relevant timescales. 

Let's see what the immediate output of the DST looks like. We will use a kernel function that looks like $`\mathcal{K}(n) \sim |n - n_0|^{-\theta}`$.


```python
from discrete_shocklets import kernel_functions, shocklets
windows = np.linspace(10, 1000, 100)  # 100 windows, equally spaced from width 10 to 1000
kernel = kernel_functions.power_cusp  # a symmetric power-law type cusp
k_args = [3.]  # arguments for the kernel; in this case, it's the parameter $\theta = 3$.
reflection = 2  # reflect the kernel over the horizontal axis

dst, largest_kernel = shocklets.cusplet(x, windows, kernel_func=kernel, kernel_args=k_args, reflection=reflection)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))
ax0, ax1 = axes
ax0.plot(x, 'darkgray')
ax0.set_xlim(0, len(x))
im = ax1.imshow(dst,
              aspect='auto',
              cmap=plt.cm.magma)

ax0.set_title('Figure 1', fontsize=15);
plt.tight_layout()
```


![png](./example/output_4_0.png)


Lighter colors indicate large positive values while darker colors indicate large negative values. We see that the immediate output of the DST captures pieces of the time series that sort of "look" like the kernel, which in this case is an upside-down spike-y kind of thing (`cusplets.power_cusp` reflected over the horizontal axis).

From Figure 1 we can also see that the DST captures this behavior over all timescales. There are similar spikes in DST intensity for the small-ish peak near $`n = 250`$ as for the much larger ones that peaks near $`n = 2000`$ and $`n = 3500`$.

### Thresholding procedures

Now we can post-process. Let's find out in what actual windows of time $`x_n`$ 
had this sort of cusp-y behavior.


```python
# extrema are the estimated extrema of the hypothesized underlying dynamics
# indicator is the cusplet indicator function, plotted below in a blue curve
# geval are indices of the cusplet indicator function where it exceeds geval
# b is a sensitivity parameter; higher b is less sensitive. 
extrema, indicator, gepoints = cusplets.classify_cusps( dst, b=0.5, geval=0.5 )
# now we can get and plot contiguous windows of cusp-y behavior
# these are created from the gepoints 
windows = cusplets.make_components(gepoints)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
ax, ax2 = axes

ax.plot(x, 'darkgray')
ax.vlines(extrema, min(x), max(x))
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_xlim(0, len(x))

# annotate the windows
for window in windows:
    ax.vlines(window, min(x), max(x), alpha=0.01, color='r')

ax2.plot(indicator, 'b-')
ax2.vlines(extrema, min(indicator), max(indicator))
ax2.set_xlim(0, len(x))

ax.set_ylabel('$x_n$')
ax2.set_xlabel('$n$')
ax2.set_ylabel('$C(n)$')

ax.set_title('Figure 2', fontsize=15)
```

![png](./example/output_7_1.png)


We'll work from the bottom up. The blue curve in the bottom panel displays the cusp indicator function, which (usually, unless you specified any custom weightings of the discrete shocklet transform) is roughly equivalent to 
`cusplets.zero_norm( np.sum(cc, axis=0) )`. The vertical bars denote relative maxima $`C^*`$ of the cusp indicator function that satisfy the additional condition
```math
C^* \geq \mu_C + b \sigma_C,
```
where $`\mu_C`$ and $`\sigma_C`$ are the mean and standard deviation of the cusp indicator function and $b$ is a tunable parameter that adjusts the sensitivity of the thresholding.
The top window again displays the $`C^*`$ in vertical bars and windows where the cusp indicator function exceeds some other threshold $`b'`$.
