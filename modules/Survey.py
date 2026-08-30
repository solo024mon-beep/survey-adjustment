"""
Survey.py

Traverse-adjustment helpers:
  - traverse_tables()  : builds the traverse observation table
  - design_matrix()    : builds the design matrix (A) and observation
                          vector (L) for least-squares adjustment
  - display_matrix()   : shared pretty-printer for (A, L)

Every function both PRINTS a formatted table (unless verbose=False) and
RETURNS its data, so results can be reused downstream (e.g. passed into
control_point_reducer.remove_control_points, or straight into a
least-squares solve) without re-parsing printed text.

Bearing equation: -EiCi + NiSi + Ei+1Ci - Ni+1Si = Bo-Bc
Distance equation: -Ei*Sin + Ni... (see design_matrix for full derivation)
Ci = cos(Bi)/(Lo*sin1")    Si = sin(Bi)/(Lo*sin1")
"""

import numpy as np
import pandas as pd
from modules.bearing import *


# ──────────────────────────────────────────────────────────────────────────
# Traverse table
# ──────────────────────────────────────────────────────────────────────────
def traverse_tables(Lines: list[str], Obs_bearing: list[str], Cal_bearing: list[str],
                     Obs_Dist: list[float], Cal_Dist: list[float],
                     verbose: bool = True) -> pd.DataFrame:
    """
    Build the traverse observation-equation table for a set of survey lines.

    Parameters
    ----------
    Lines : list[str]
        Two-letter station codes for each line, e.g. 'AX'.
    Obs_bearing : list[str]
        Observed bearings as DMS strings, e.g. '338 18 24.58'.
    Cal_bearing : list[str]
        Calculated (adjusted) bearings as DMS strings.
    Obs_Dist : list[float]
        Observed distances.
    Cal_Dist : list[float]
        Calculated (adjusted) distances.
    verbose : bool, default True
        Print the formatted table. Set False to build silently.

    Returns
    -------
    pd.DataFrame
        Columns: From, To, Cal_Bearing, Obs_Bearing, Cal_Distance,
        Obs_Distance, CosBi, SinBi, Ci, Si, Lo - Lc, Bo - Bc
    """
    From = [line[0] for line in Lines]
    To = [line[1] for line in Lines]

    Obs_Dist_f = list(map(float, Obs_Dist))
    Cal_Dist_f = list(map(float, Cal_Dist))

    df = pd.DataFrame({
        'From':         From,
        'To':           To,
        'Cal_Bearing':  Cal_bearing,
        'Obs_Bearing':  Obs_bearing,
        'Cal_Distance': Cal_Dist_f,
        'Obs_Distance': Obs_Dist_f,
    })

    # --- CosBi and SinBi (full precision for Ci/Si, rounded for display) ---
    CosBi_full = [np.cos(np.radians(dms_to_decimal(b))) for b in Obs_bearing]
    SinBi_full = [np.sin(np.radians(dms_to_decimal(b))) for b in Obs_bearing]

    df['CosBi'] = [round(v, 3) for v in CosBi_full]
    df['SinBi'] = [round(v, 3) for v in SinBi_full]

    sin1sec = np.sin(np.radians(1 / 3600))

    df['Ci'] = [round(c / (d * sin1sec), 3) for c, d in zip(CosBi_full, Obs_Dist_f)]
    df['Si'] = [round(s / (d * sin1sec), 3) for s, d in zip(SinBi_full, Obs_Dist_f)]
    df['Lo - Lc'] = [round(o - c, 3) for o, c in zip(Obs_Dist_f, Cal_Dist_f)]
    df['Bo - Bc'] = [dif_DMS(o, c) for o, c in zip(Obs_bearing, Cal_bearing)]

    if verbose:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.colheader_justify', 'center')

        print("\n" + "=" * 100)
        print(" " * 35 + "TRAVERSE TABLE")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")

    return df


# ──────────────────────────────────────────────────────────────────────────
# Design matrix / observation vector
# ──────────────────────────────────────────────────────────────────────────
def design_matrix(Lines: list[str], obs_bearing: list[str], cal_bearing: list[str],
                   obs_dist: list[float], cal_dist: list[float],
                   observation_types: tuple = ('distance', 'bearing'),
                   fixed_stations: set | list | None = None,
                   verbose: bool = True):
    """
    Build the design matrix (A) and observation vector (L) for a
    least-squares traverse adjustment.

    Unknowns are ordered station-by-station as [δE, δN] for each
    FREE station found in `Lines` (i.e. not in `fixed_stations`),
    sorted alphabetically.

    Parameters
    ----------
    Lines : list[str]
        Two-letter station codes for each line, e.g. 'AX'.
    obs_bearing : list[str]
        Observed bearings as DMS strings.
    cal_bearing : list[str]
        Calculated (adjusted) bearings as DMS strings.
    obs_dist : list[float]
        Observed distances.
    cal_dist : list[float]
        Calculated (adjusted) distances.
    observation_types : tuple, default ('distance', 'bearing')
        Which observation type(s) to include as rows. Use
        `('distance',)` for a distance-only adjustment, or
        `('bearing',)` for a bearing/direction-only adjustment.
        NOTE: a distance-only adjustment can't recover orientation, and
        a bearing-only adjustment can't recover scale -- for either to
        be solvable you generally need to hold enough stations fixed
        via `fixed_stations` (this function warns if the resulting A
        looks column-rank-deficient).
    fixed_stations : set | list, optional
        Station codes to hold fixed (control points). Their δE/δN
        columns are omitted entirely -- they contribute to a line's
        row the same way any station does, they just aren't solved
        for. Defaults to none (every station in `Lines` is free/unknown,
        matching the original behavior).
    verbose : bool, default True
        Print the formatted A and L tables. Set False to build silently.

    Returns
    -------
    A          : np.ndarray  shape (n_rows, 2*n_free_stations)
    L          : np.ndarray  shape (n_rows,)
    stations   : list[str]   FREE station codes, in the order used for columns
    col_labels : list[str]   e.g. ['δEA', 'δNA', 'δEB', 'δNB', ...] (free stations only)
    row_labels : list[str]   e.g. ['Dist_AB', 'Bear_AB', ...] (only the
                              included observation_types)
    """
    valid_types = {'distance', 'bearing'}
    observation_types = tuple(observation_types)
    bad = set(observation_types) - valid_types
    if bad:
        raise ValueError(f"observation_types can only contain {valid_types}, "
                          f"got {bad}")
    if not observation_types:
        raise ValueError("observation_types can't be empty -- include at "
                          "least 'distance' or 'bearing'.")

    fixed_stations = set(fixed_stations) if fixed_stations else set()

    sin1sec = np.sin(np.radians(1 / 3600))

    # Guard against string input
    obs_dist = list(map(float, obs_dist))
    cal_dist = list(map(float, cal_dist))

    SinBi, CosBi, Ci, Si = [], [], [], []
    Lo_Lc, Bo_Bc = [], []

    for i in range(len(Lines)):
        rad = np.radians(dms_to_decimal(obs_bearing[i]))
        s = np.sin(rad)
        c = np.cos(rad)
        SinBi.append(s)
        CosBi.append(c)
        Ci.append(c / (obs_dist[i] * sin1sec))
        Si.append(s / (obs_dist[i] * sin1sec))
        Lo_Lc.append(obs_dist[i] - cal_dist[i])
        Bo_Bc.append((dms_to_decimal(obs_bearing[i]) - dms_to_decimal(cal_bearing[i])) * 3600)

    # ── Derive stations dynamically from the Lines actually used ───────────
    all_stations = sorted(set(''.join(Lines)))
    unknown_in_fixed = fixed_stations - set(all_stations)
    if unknown_in_fixed:
        raise ValueError(f"fixed_stations contains station(s) not present "
                          f"in Lines: {unknown_in_fixed}")

    stations = [s for s in all_stations if s not in fixed_stations]  # free only
    idx = {s: i * 2 for i, s in enumerate(stations)}  # δE index; δN = idx+1

    n_unknowns = len(stations) * 2
    n_lines = len(Lines)
    rows_per_line = len(observation_types)

    A = np.zeros((rows_per_line * n_lines, n_unknowns))
    L = np.zeros(rows_per_line * n_lines)
    row_labels = []

    for i, line in enumerate(Lines):
        fr, to = line[0], line[1]
        S, C = SinBi[i], CosBi[i]
        ci, si = Ci[i], Si[i]

        # Column indices only exist for FREE stations; a fixed station's
        # term is simply omitted (its known/exact position doesn't need a
        # correction), same idea as a control point in a leveling network.
        e_i, n_i = idx.get(fr), (idx[fr] + 1 if fr in idx else None)
        e_j, n_j = idx.get(to), (idx[to] + 1 if to in idx else None)

        for obs_type in observation_types:
            row = len(row_labels)
            if obs_type == 'distance':
                # -δEi·Sin - δNi·Cos + δEj·Sin + δNj·Cos = Lo-Lc
                if e_i is not None:
                    A[row, e_i] = -S
                    A[row, n_i] = -C
                if e_j is not None:
                    A[row, e_j] = S
                    A[row, n_j] = C
                L[row] = Lo_Lc[i]
                row_labels.append(f'Dist_{line}')
            else:  # 'bearing'
                # -δEi·Ci + δNi·Si + δEj·Ci - δNj·Si = Bo-Bc
                if e_i is not None:
                    A[row, e_i] = -ci
                    A[row, n_i] = si
                if e_j is not None:
                    A[row, e_j] = ci
                    A[row, n_j] = -si
                L[row] = Bo_Bc[i]
                row_labels.append(f'Bear_{line}')

    col_labels = [f'δ{axis}{s}' for s in stations for axis in ('E', 'N')]

    # ── Datum-defect warning ────────────────────────────────────────────
    # A distance-only network can't recover orientation, and a
    # bearing-only network can't recover scale, unless enough stations
    # are held fixed. Catch this early with a column-rank check rather
    # than letting a downstream np.linalg.inv fail with a bare
    # LinAlgError.
    if n_unknowns > 0:
        rank = np.linalg.matrix_rank(A)
        if rank < n_unknowns and verbose:
            print(f"\n⚠ WARNING: design matrix A has column rank {rank} but "
                  f"{n_unknowns} unknowns ({len(stations)} free station(s)) "
                  f"-- the normal matrix A'PA will be SINGULAR. This "
                  f"usually means not enough stations are held fixed for "
                  f"{observation_types} alone to determine the network "
                  f"(e.g. distance-only can't fix orientation, "
                  f"bearing-only can't fix scale). Add more fixed_stations "
                  f"or include both observation types.")

    if verbose:
        display_matrix(A, L, col_labels, row_labels, title="DESIGN MATRIX (A)")

    return A, L, stations, col_labels, row_labels


# ──────────────────────────────────────────────────────────────────────────
# Weight matrix
# ──────────────────────────────────────────────────────────────────────────
def weight_matrix(Lines: list[str], dist_std, bearing_std, sigma0: float = 1.0,
                   observation_types: tuple = ('distance', 'bearing'),
                   verbose: bool = True):
    """
    Build the diagonal weight matrix (P) for the observations from their
    standard deviations, in the same row order as L from design_matrix().

    Uses the standard inverse-variance relation:

        P_ii = (sigma0 / sigma_i) ** 2

    where sigma0 is the a-priori reference standard deviation of unit
    weight (defaults to 1.0) and sigma_i is the standard deviation of
    observation i.

    Parameters
    ----------
    Lines : list[str]
        Two-letter station codes for each line, e.g. 'AX'. Must match
        the `Lines` passed to design_matrix() so the rows line up.
    dist_std : float | list[float]
        Standard deviation(s) of the distance observations, in the same
        units as your distances. A single number is applied to every
        line; a list must have one entry per line, in the same order
        as `Lines`. Ignored if 'distance' isn't in `observation_types`.
    bearing_std : float | list[float]
        Standard deviation(s) of the bearing/angle observations, in
        arc-seconds (matching the units of Bo-Bc in design_matrix).
        Same broadcasting rules as dist_std. Ignored if 'bearing' isn't
        in `observation_types`.
    sigma0 : float, default 1.0
        A-priori reference standard deviation of unit weight.
    observation_types : tuple, default ('distance', 'bearing')
        Which observation type(s) to include -- MUST match whatever was
        passed to design_matrix() so P lines up with A and L.
    verbose : bool, default True
        Print the formatted weight matrix. Set False to build silently.

    Returns
    -------
    P          : np.ndarray  diagonal, shape (rows_per_line*n_lines, same)
    row_labels : list[str]   e.g. ['Dist_AB', 'Bear_AB', ...] (only the
                              included observation_types, same order as
                              design_matrix's L, so P lines up with A/L)
    """
    valid_types = {'distance', 'bearing'}
    observation_types = tuple(observation_types)
    bad = set(observation_types) - valid_types
    if bad:
        raise ValueError(f"observation_types can only contain {valid_types}, "
                          f"got {bad}")
    if not observation_types:
        raise ValueError("observation_types can't be empty -- include at "
                          "least 'distance' or 'bearing'.")

    n_lines = len(Lines)

    def _broadcast(std, name):
        if isinstance(std, (int, float)):
            std = [float(std)] * n_lines
        std = list(std)
        if len(std) != n_lines:
            raise ValueError(f"{name} must be a scalar or a list of length "
                              f"{n_lines} (got {len(std)})")
        std = [float(x) for x in std]
        if any(x <= 0 for x in std):
            raise ValueError(f"{name} must contain only positive standard deviations")
        return std

    ds = _broadcast(dist_std, "dist_std") if 'distance' in observation_types else None
    bs = _broadcast(bearing_std, "bearing_std") if 'bearing' in observation_types else None

    rows_per_line = len(observation_types)
    diag = np.zeros(rows_per_line * n_lines)
    row_labels = []
    for i, line in enumerate(Lines):
        for obs_type in observation_types:
            row = len(row_labels)
            if obs_type == 'distance':
                diag[row] = (sigma0 / ds[i]) ** 2
                row_labels.append(f'Dist_{line}')
            else:
                diag[row] = (sigma0 / bs[i]) ** 2
                row_labels.append(f'Bear_{line}')

    P = np.diag(diag)

    if verbose:
        df_P = pd.DataFrame(P, index=row_labels, columns=row_labels)
        pd.set_option('display.float_format', '{:>10.3f}'.format)
        pd.set_option('display.width', 160)

        print("\n" + "=" * 90)
        print(" " * 30 + "WEIGHT MATRIX (P)")
        print("=" * 90)
        print(df_P.to_string())
        print("=" * 90 + "\n")

    return P, row_labels


# ──────────────────────────────────────────────────────────────────────────
# Shared pretty-printer
# ──────────────────────────────────────────────────────────────────────────
def display_matrix(A: np.ndarray, L: np.ndarray, col_labels: list[str],
                    row_labels: list[str], title: str = "DESIGN MATRIX") -> None:
    """Print a design matrix A and observation vector L as formatted tables."""
    df_A = pd.DataFrame(A, index=row_labels, columns=col_labels)
    df_L = pd.DataFrame(L, index=row_labels, columns=['L (obs-cal)'])

    pd.set_option('display.float_format', '{:>10.3f}'.format)
    pd.set_option('display.width', 160)

    print("\n" + "=" * 90)
    print(" " * 25 + title)
    print("=" * 90)
    print(df_A.to_string())
    print()
    print("=" * 40)
    print("  OBSERVATION VECTOR  (L)")
    print("=" * 40)
    print(df_L.to_string())
    print("=" * 40 + "\n")
