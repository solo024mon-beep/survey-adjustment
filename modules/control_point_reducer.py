"""
control_point_reducer.py

Interactively builds ONE design matrix (A), ONE observation vector (L),
and ONE weight matrix (P) for a traverse least-squares adjustment --
already filtered to:
  - only the observation type(s) you choose (distance-only, bearing-only,
    or both), and
  - only the FREE (unknown) stations, i.e. control points never get
    columns in the first place.

This relies on Survey.design_matrix()'s `observation_types` and
`fixed_stations` parameters to do the filtering/reduction directly, so
there is no "build full matrix, then strip columns" step -- A, L, and P
are each built exactly once and are guaranteed to line up with each
other (same row order, same column order).
"""

import numpy as np
from .Survey import design_matrix, weight_matrix


# ──────────────────────────────────────────────────────────────────────────
# Helper: build a correctly-ordered provisional-values vector
# ──────────────────────────────────────────────────────────────────────────
def build_prov_values(unknown_stations: list[str], prov_coords: dict) -> list:
    """
    Build the provisional-values vector for least_squares(), in the same
    [E, N] per-station order that design_matrix() uses for its columns
    (col_labels: 'δEA', 'δNA', 'δEB', 'δNB', ...).

    Parameters
    ----------
    unknown_stations : list[str]
        The `stations` list returned by design_matrix() / interactive_reduce().
    prov_coords : dict[str, tuple[float, float]]
        Provisional (approximate) coordinates for every station in
        `unknown_stations`, as {station: (E, N)}.

    Returns
    -------
    list[float]
        [E1, N1, E2, N2, ...] ready to pass as prov_values to least_squares().
    """
    missing = [s for s in unknown_stations if s not in prov_coords]
    if missing:
        raise ValueError(f"Missing provisional coordinates for stations: {missing}")

    prov_values = []
    for s in unknown_stations:
        E, N = prov_coords[s]
        prov_values.extend([E, N])
    return prov_values


# ──────────────────────────────────────────────────────────────────────────
# Full interactive workflow
# ──────────────────────────────────────────────────────────────────────────
def interactive_reduce(Lines: list[str], obs_bearing: list[str], cal_bearing: list[str],
                        obs_dist: list[float], cal_dist: list[float],
                        dist_std=None, bearing_std=None):
    """
    1. Ask which observation type(s) to use: distance-only, bearing-only,
       or both.
    2. Ask which station(s) are control points (fixed / known).
    3. Get the standard deviation(s) needed to build the weight matrix P.
       If `dist_std`/`bearing_std` are passed in (a scalar applied to
       every line, or a list with one entry per line in `Lines`, exactly
       like Survey.weight_matrix() accepts), those are used directly and
       the corresponding prompt is skipped. Otherwise the user is
       prompted for a single scalar std dev at runtime. Only whichever
       std dev is actually needed for the observation types chosen in
       step 1 is required.
    4. Build A, L, and P -- ONCE, already filtered/reduced -- and
       display them.

    Parameters
    ----------
    dist_std : float | list[float], optional
        Standard deviation(s) of the distance observations, already
        known in code (e.g. field-survey spec sheet values, one per
        line in `Lines`). If omitted, prompts for a single value.
        Ignored if 'distance' isn't part of the chosen observation types.
    bearing_std : float | list[float], optional
        Same idea, for the bearing observations (arc-seconds). Ignored
        if 'bearing' isn't part of the chosen observation types.

    Returns
    -------
    A          : np.ndarray  design matrix, control-point columns already excluded
    L          : np.ndarray  observation vector, filtered to the chosen observation type(s)
    P          : np.ndarray  diagonal weight matrix, same row order as A/L
    unknown_stations : list[str]  free station codes, in column order
    col_labels : list[str]
    row_labels : list[str]
    """
    all_stations = sorted(set(''.join(Lines)))
    print(f"Stations found in network: {', '.join(all_stations)}")

    # ── 1. Observation type ────────────────────────────────────────────
    print("\nWhich observation type(s) should be used?")
    print("  [1] distance only")
    print("  [2] bearing only")
    print("  [3] both (default)")
    choice = input("Enter 1, 2, or 3: ").strip()
    obs_type_map = {
        '1': ('distance',),
        '2': ('bearing',),
        '3': ('distance', 'bearing'),
        '':  ('distance', 'bearing'),
    }
    if choice not in obs_type_map:
        raise ValueError("Invalid choice -- enter 1, 2, or 3.")
    observation_types = obs_type_map[choice]

    # ── 2. Control points ───────────────────────────────────────────────
    raw = input("Enter control point station code(s) (space or comma separated): ")
    control_points = [c for c in raw.replace(',', ' ').upper().split() if c]

    invalid = [c for c in control_points if c not in all_stations]
    if invalid:
        raise ValueError(f"Not valid stations in this network: {invalid}")
    if len(control_points) == len(all_stations):
        raise ValueError("At least one station must remain unknown — "
                          "you can't mark every station as control.")

    # ── 3. Standard deviations for the weight matrix ───────────────────
    # Use whatever was passed in from the code; only prompt for what's
    # missing and actually needed.
    if 'distance' in observation_types and dist_std is None:
        dist_std = float(input("Standard deviation of distance observations "
                                "(same units as distances): "))
    if 'bearing' in observation_types and bearing_std is None:
        bearing_std = float(input("Standard deviation of bearing observations "
                                   "(arc-seconds): "))

    # ── 4. Build A, L, P exactly once ───────────────────────────────────
    A, L, unknown_stations, col_labels, row_labels = design_matrix(
        Lines, obs_bearing, cal_bearing, obs_dist, cal_dist,
        observation_types=observation_types,
        fixed_stations=control_points,
        verbose=True,
    )

    P, p_row_labels = weight_matrix(
        Lines, dist_std, bearing_std,
        observation_types=observation_types,
        verbose=True,
    )
    assert p_row_labels == row_labels, "P row order doesn't match A/L row order"

    # Rows that end up all-zero only tie together control points -> no
    # info left about the remaining unknowns. Flag so they (and the
    # matching row/col of P) can be dropped before running least squares.
    dropped_rows = [i for i in range(A.shape[0]) if np.allclose(A[i], 0)]
    if dropped_rows:
        print("\n⚠  These observation rows involve ONLY control points and carry")
        print("   no information about the remaining unknowns:")
        for r in dropped_rows:
            print(f"     - {row_labels[r]}")
        print("   Consider deleting these rows from A, L, AND the matching")
        print("   row/column of P before running least squares.\n")

    print(f"Observation type(s) used : {', '.join(observation_types)}")
    print(f"Control points removed   : {', '.join(control_points)}")
    print(f"Remaining unknowns       : {', '.join(unknown_stations)}")

    return A, L, P, unknown_stations, col_labels, row_labels
