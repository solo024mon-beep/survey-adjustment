"""
errorellipse.py

Computes the standard error ellipse for one or more points from their
cofactor matrix (Qxx), e.g. result['Qxx'] from leastsquares.least_squares().

Formulas (matching the hand-worked solution)
----------------------------------------------
For a point's 2x2 cofactor sub-block:

    [ qxx  qxy ]
    [ qxy  qyy ]

    w      = sqrt((qxx - qyy)^2 + 4*qxy^2)
    a      = S0 * sqrt( 1/2 * (qxx + qyy + w) )   semi-major axis
    b      = S0 * sqrt( 1/2 * (qxx + qyy - w) )   semi-minor axis
    Sx     = S0 * sqrt(qxx)                       std. error along E
    Sy     = S0 * sqrt(qyy)                       std. error along N
    tan(2t) = 2*qxy / (qxx - qyy)
    t      = 1/2 * atan2(2*qxy, qxx - qyy)        orientation, from E axis
    Bearing = t converted to a 0-360 deg surveying bearing (clockwise
              from North), correcting for the quadrant of (qxy, qxx-qyy)

S0 (sigma0) is the single scale multiplier applied throughout -- pass
whatever combination of std. deviation of unit weight and confidence-level
factor you want baked into it (e.g. S0 = sigma0 * sqrt(chi2.ppf(p, df=2))
if you want a p-confidence ellipse; S0 = sigma0 alone for the standard
1-sigma ellipse).
"""

import numpy as np
import pandas as pd


def error_ellipse(Qxx: np.ndarray, S0: float = 1.0,
                   point_labels: list[str] = None, verbose: bool = True,
                   sd: float = None, sigma0: float = None) -> pd.DataFrame:
    """
    Compute the error ellipse for one or more points.

    Parameters
    ----------
    Qxx : np.ndarray
        Either the full cofactor matrix of unknowns for a network with
        unknowns ordered [E1, N1, E2, N2, ...] (shape (2n, 2n)), or a
        single point's 2x2 cofactor sub-block (shape (2, 2)). Only the
        block-diagonal 2x2 sub-matrices are used -- cross-terms between
        different points' unknowns don't affect each point's own error
        ellipse.
    S0 : float, default 1.0
        The single scale multiplier used in every formula above (called
        S0/sigma0 in the hand-worked solution). Fold in whatever you need
        -- standard deviation of unit weight, a confidence-level factor,
        or both.
    point_labels : list[str], optional
        Label for each point (e.g. station codes ['A', 'B', 'C', ...]).
        Defaults to '1', '2', '3', ... if omitted. Must have exactly
        Qxx.shape[0] // 2 entries.
    verbose : bool, default True
        Print a formatted results table. Set False to compute silently.
    sd, sigma0 : float, optional
        Aliases for `S0` (older parameter names). If given, they override
        `S0`.

    Returns
    -------
    pd.DataFrame
        Columns: Point, w, a, b, Sx, Sy, theta_deg, Bearing.
        w = intermediate radical term; a/b = semi-major/minor axis
        lengths; Sx/Sy = standard error along the E/N axes directly (not
        rotated); theta_deg = orientation of the major axis in degrees,
        measured mathematically from the E axis; Bearing = orientation
        converted to a surveying bearing (0-360 deg, clockwise from
        North).
    """
    if sd is not None:
        S0 = sd
    if sigma0 is not None:
        S0 = sigma0

    Qxx = np.asarray(Qxx, dtype=float)

    if Qxx.ndim != 2 or Qxx.shape[0] != Qxx.shape[1]:
        raise ValueError(f"Qxx must be a square matrix, got shape {Qxx.shape}")
    if Qxx.shape[0] % 2 != 0:
        raise ValueError(f"Qxx must have an even dimension (2 rows/cols per "
                          f"point: E, N), got shape {Qxx.shape}")

    n_points = Qxx.shape[0] // 2

    if point_labels is None:
        point_labels = [str(i + 1) for i in range(n_points)]
    if len(point_labels) != n_points:
        raise ValueError(f"point_labels must have {n_points} entries, "
                          f"got {len(point_labels)}")

    results = []

    for i in range(n_points):
        # Extract THIS point's own 2x2 cofactor sub-block.
        q_block = Qxx[2 * i:2 * i + 2, 2 * i:2 * i + 2]
        qxx, qyy, qxy = q_block[0, 0], q_block[1, 1], q_block[0, 1]

        # w = sqrt((qxx - qyy)^2 + 4*qxy^2)
        w = np.sqrt((qxx - qyy) ** 2 + 4 * qxy ** 2)

        # a = S0 * sqrt(1/2 * (qxx + qyy + w))
        # b = S0 * sqrt(1/2 * (qxx + qyy - w))
        # qxx/qyy are allowed to be negative (not a valid cofactor block,
        # but the formula itself doesn't break) -- if 1/2*(qxx+qyy-w) is
        # negative, b comes back NaN instead of crashing the whole batch.
        with np.errstate(invalid='ignore'):
            a = S0 * np.sqrt(0.5 * (qxx + qyy + w))
            b = S0 * np.sqrt(0.5 * (qxx + qyy - w))

        # Sx = S0 * sqrt(qxx), Sy = S0 * sqrt(qyy)
        # NaN (with a RuntimeWarning) if qxx/qyy is negative -- expected,
        # not a bug: Sx/Sy just aren't defined for a negative variance.
        with np.errstate(invalid='ignore'):
            Sx = S0 * np.sqrt(qxx)
            Sy = S0 * np.sqrt(qyy)

        # tan(2t) = 2*qxy / (qxx - qyy)  ->  t = 1/2 * atan2(2*qxy, qxx-qyy)
        # atan2 (rather than plain atan of the ratio) keeps qxy's sign and
        # picks the correct quadrant automatically.
        theta = 0.5 * np.arctan2(2 * qxy, qxx - qyy)
        theta_deg = np.degrees(theta)

        # Convert to a surveying bearing (0-360 deg, clockwise from
        # North). These branches correct for the +/-180 degree
        # ambiguity in the principal-axis direction, matching the
        # quadrant check worked by hand. The final `else` handles
        # qxy == 0 and/or qxx == qyy (axes aligned with E/N, or a
        # perfect circle where orientation is arbitrary).
        if 2 * qxy > 0 and qxx - qyy > 0:
            bearing = np.mod(90 - (theta_deg + 90), 360)
        elif 2 * qxy > 0 and qxx - qyy < 0:
            bearing = np.mod(90 - (theta_deg + 180), 360)
        elif 2 * qxy < 0 and qxx - qyy < 0:
            bearing = np.mod(90 - (theta_deg + 270), 360)
        elif 2 * qxy < 0 and qxx - qyy > 0:
            bearing = np.mod(90 - (theta_deg + 360), 360)
        else:
            bearing = np.mod(90 - theta_deg, 360)

        results.append({
            'Point': point_labels[i],
            'w': w,
            'a': a,
            'b': b,
            'Sx': Sx,
            'Sy': Sy,
            'theta_deg': theta_deg,
            'Bearing': bearing,
        })

    df = pd.DataFrame(results)

    if verbose:
        pd.set_option('display.float_format', '{:>10.4f}'.format)
        pd.set_option('display.width', 120)
        print("\n" + "=" * 80)
        print(" " * 25 + "ERROR ELLIPSE")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80 + "\n")

    return df
