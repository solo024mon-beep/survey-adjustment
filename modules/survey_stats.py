"""
survey_stats.py

Descriptive statistics and outlier screening for a set of repeated
survey observations (e.g. repeated distance or angle measurements).

Follows the standard "probable error" framework used in surveying:
E50, E90, and E99 give the interval within which 50%, 90%, and 99% of
observations are expected to fall for a normally-distributed dataset;
anything outside the E99 interval is flagged as a likely outlier.
"""

import statistics as stat
import math


def _interval_check(data: tuple, mean: float, half_width: float,
                     label: str, verbose: bool = True) -> dict:
    """
    Count how many observations fall within [mean - half_width, mean + half_width]
    and report the ones that do (and the percentage of the dataset they represent).

    Returns
    -------
    dict with keys: 'interval', 'within', 'count', 'percent'
    """
    lower, upper = mean - half_width, mean + half_width
    within = [x for x in data if lower < x < upper]
    percent = len(within) / len(data) * 100

    if verbose:
        print(f'{label}: {half_width:.4f}')
        print(f'interval: from {lower:.2f} to {upper:.2f}')
        for x in within:
            print(f'{x:.2f} is within the interval')
        print(f'number of observations within the interval: {len(within)}')
        print(f'percentage: {percent:.3f}%')
        print('_' * 25)

    return {'interval': (lower, upper), 'within': within,
            'count': len(within), 'percent': percent}


def statistics(data: tuple, verbose: bool = True) -> dict:
    """
    Compute descriptive statistics and probable-error outlier screening
    for a set of repeated observations.

    Parameters
    ----------
    data : tuple[int | float]
        The repeated observations.
    verbose : bool, default True
        Print the formatted report. Set False to compute silently.

    Returns
    -------
    dict with keys:
        'n'                : number of observations
        'mean'              : arithmetic mean
        'median'            : median
        'mode'              : list of mode(s)
        'range'             : max - min
        'variance'          : sample variance
        'std_dev'           : sample standard deviation
        'std_err_of_mean'   : standard error of the mean, std_dev / sqrt(n)
        'E50', 'E90', 'E99' : probable-error half-widths (dicts, see _interval_check)
        'outliers'          : list of observations outside the E99 (99%) interval
    """
    n = len(data)
    mean = stat.mean(data)
    median = stat.median(data)
    mode = stat.multimode(data)
    data_range = max(data) - min(data)
    variance = stat.variance(data, xbar=mean)
    std_dev = stat.stdev(data, xbar=mean)
    std_err_of_mean = std_dev / math.sqrt(n)

    if verbose:
        print(f'total number of observations: {n}')
        print(f'mean: {mean:.4f}')
        print(f'median: {median:.4f}')
        print(f'mode: {mode}')
        print(f'range: {data_range:.4f}')
        print(f'variance: {variance:.4f}')
        print(f'standard deviation: \u2213{std_dev:.4f}')
        print(f'standard error of the mean: {std_err_of_mean:.4f}')

    # ── Probable-error intervals ─────────────────────────────────────────
    # Multipliers are z-scores for the stated confidence level under a
    # normal distribution; adjust to match your assignment's specification
    # if it calls for different confidence levels.
    if verbose:
        print('---------------E50----------------')
    e50 = _interval_check(data, mean, std_dev * 0.675, 'e50', verbose)

    if verbose:
        print('---------------E90----------------')
    e90 = _interval_check(data, mean, std_dev * 1.645, 'e90', verbose)

    if verbose:
        print('----------REJECTION CRITERIA----------')
    e99 = _interval_check(data, mean, std_dev * 2.576, 'e99', verbose)

    # ── Outlier test ─────────────────────────────────────────────────────
    # An observation more than Rc (= the E99 half-width) from the mean is
    # flagged as an outlier -- equivalent to "outside the E99 interval".
    Rc = std_dev * 2.576
    outliers = [x for x in data if abs(x - mean) > Rc]

    if verbose:
        print('----------OUTLIER TEST-----------')
        if outliers:
            for x in outliers:
                print(f'{x:.2f} is an outlier')
        else:
            print('no outliers found')
        print(f'number of outliers: {len(outliers)}')

    return {
        'n': n,
        'mean': mean,
        'median': median,
        'mode': mode,
        'range': data_range,
        'variance': variance,
        'std_dev': std_dev,
        'std_err_of_mean': std_err_of_mean,
        'E50': e50,
        'E90': e90,
        'E99': e99,
        'outliers': outliers,
    }


def _r_strength(r: float) -> str:
    """
    Rough, commonly-used rule-of-thumb label for |r|. Not a statistical
    test -- just a convenient description for a report.
    """
    a = abs(r)
    if a >= 0.9:
        band = 'very strong'
    elif a >= 0.7:
        band = 'strong'
    elif a >= 0.5:
        band = 'moderate'
    elif a >= 0.3:
        band = 'weak'
    else:
        band = 'very weak / negligible'
    sign = 'positive' if r >= 0 else 'negative'
    return f'{band} {sign} correlation'


def correlation(x: tuple, y: tuple, verbose: bool = True) -> dict:
    """
    Pearson correlation coefficient and line-of-best-fit (simple linear
    regression) for paired observations (x, y) -- e.g. two survey
    quantities you want to check the relationship between.

    Follows the by-hand method: deviations from each mean (Rx, Ry), their
    products and squares, covariance, Sx/Sy (sample std. devs of x and
    y), Pearson's r = covariance / (Sx * Sy), and the best-fit line
    y = a*x + b with a = r*Sy/Sx and b = ybar - a*xbar.

    Parameters
    ----------
    x, y : tuple[int | float]
        Paired observations. Must be the same length (and length >= 2).
    verbose : bool, default True
        Print the formatted per-observation table and summary. Set False
        to compute silently.

    Returns
    -------
    dict with keys:
        'n'                     : number of paired observations
        'x_mean', 'y_mean'      : means of x and y
        'Rx', 'Ry'              : lists of deviations from the mean
        'RxRy', 'Rx2', 'Ry2'    : lists of per-observation products/squares
        'sum_RxRy', 'sum_Rx2', 'sum_Ry2' : their sums
        'covariance'            : sum_RxRy / (n - 1)
        'Sx', 'Sy'              : sample standard deviations of x and y
        'r'                     : Pearson correlation coefficient
        'r_squared'             : coefficient of determination (r^2)
        'strength'              : plain-language description of |r|
        'a', 'b'                : best-fit line slope and intercept
        'predict'               : callable, predict(x_new) -> a*x_new + b
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got "
                          f"{len(x)} and {len(y)}")
    n = len(x)
    if n < 2:
        raise ValueError(f"Need at least 2 paired observations, got {n}")

    x_mean = stat.mean(x)
    y_mean = stat.mean(y)

    Rx = [xi - x_mean for xi in x]
    Ry = [yi - y_mean for yi in y]
    RxRy = [rx * ry for rx, ry in zip(Rx, Ry)]
    Rx2 = [rx ** 2 for rx in Rx]
    Ry2 = [ry ** 2 for ry in Ry]

    sum_RxRy = sum(RxRy)
    sum_Rx2 = sum(Rx2)
    sum_Ry2 = sum(Ry2)

    covariance = sum_RxRy / (n - 1)
    Sx = math.sqrt(sum_Rx2 / (n - 1))
    Sy = math.sqrt(sum_Ry2 / (n - 1))

    if Sx == 0 or Sy == 0:
        raise ValueError("x or y has zero variance (all identical values) "
                          "-- correlation and the best-fit line slope are "
                          "undefined.")

    r = covariance / (Sx * Sy)
    r_squared = r ** 2
    strength = _r_strength(r)

    a = r * Sy / Sx
    b = y_mean - a * x_mean

    def predict(x_new):
        return a * x_new + b

    if verbose:
        print(f'{"x":>10}{"y":>10}{"Rx":>10}{"Ry":>10}'
              f'{"Rx*Ry":>12}{"Rx^2":>10}{"Ry^2":>10}')
        for xi, yi, rx, ry, rxry, rx2, ry2 in zip(x, y, Rx, Ry, RxRy,
                                                   Rx2, Ry2):
            print(f'{xi:>10.4f}{yi:>10.4f}{rx:>10.4f}{ry:>10.4f}'
                  f'{rxry:>12.4f}{rx2:>10.4f}{ry2:>10.4f}')
        print('-' * 72)
        print(f'{"sums":>10}{"":>10}{"":>10}{"":>10}'
              f'{sum_RxRy:>12.4f}{sum_Rx2:>10.4f}{sum_Ry2:>10.4f}')
        print(f'x_mean: {x_mean:.4f}   y_mean: {y_mean:.4f}')
        print('_' * 25)
        print(f'covariance: {covariance:.4f}')
        print(f'Sx: \u00b1{Sx:.6f}')
        print(f'Sy: \u00b1{Sy:.6f}')
        print(f'r (Pearson): {r:.5f}   ({strength})')
        print(f'r^2: {r_squared:.5f}')
        print('_' * 25)
        print('Line of best fit: y = a*x + b')
        print(f'a: {a:.4f}')
        print(f'b: {b:.4f}')
        print(f'y = {a:.4f}x + {b:.4f}')

    return {
        'n': n,
        'x_mean': x_mean,
        'y_mean': y_mean,
        'Rx': Rx,
        'Ry': Ry,
        'RxRy': RxRy,
        'Rx2': Rx2,
        'Ry2': Ry2,
        'sum_RxRy': sum_RxRy,
        'sum_Rx2': sum_Rx2,
        'sum_Ry2': sum_Ry2,
        'covariance': covariance,
        'Sx': Sx,
        'Sy': Sy,
        'r': r,
        'r_squared': r_squared,
        'strength': strength,
        'a': a,
        'b': b,
        'predict': predict,
    }
