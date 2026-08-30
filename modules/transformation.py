import numpy as np
import math
import pandas as pd


# ============================================================
# 1. Solve Conformal Transformation Parameters
# ============================================================
def solve_conformal(E_s, N_s, E_T, N_T, verbose=True):
    """
    Solve the 2D Conformal (Helmert) Transformation using Least Squares.

    Model:  E_T = a*E_s - b*N_s + c
            N_T = b*E_s + a*N_s + d

    Parameters:
    ----------
    E_s, N_s : Source coordinates
    E_T, N_T : Target coordinates
    verbose : bool, default True
        Print the design matrix A, observation vector L, and the solved
        parameters.

    Returns:
    -------
    a, b, c, d : Transformation parameters
    scale : Scale factor
    theta : Rotation angle (degrees)
    """
    E_s = np.asarray(E_s, dtype=float)
    N_s = np.asarray(N_s, dtype=float)
    E_T = np.asarray(E_T, dtype=float)
    N_T = np.asarray(N_T, dtype=float)
    n = len(E_s)
    if not (len(N_s) == len(E_T) == len(N_T) == n):
        raise ValueError("Coordinate arrays must all have the same length.")
    if n < 2:
        raise ValueError("At least 2 control points are required "
                         "(4 parameters need at least 4 equations).")

    # Design Matrix
    A = np.zeros((2 * n, 4))
    L = np.zeros((2 * n, 1))
    for i in range(n):
        # Easting equation
        A[2 * i] = [E_s[i], -N_s[i], 1, 0]
        L[2 * i] = E_T[i]
        # Northing equation
        A[2 * i + 1] = [N_s[i], E_s[i], 0, 1]
        L[2 * i + 1] = N_T[i]

    if verbose:
        row_labels = []
        for i in range(n):
            row_labels += [f"E{i + 1}", f"N{i + 1}"]
        A_df = pd.DataFrame(A, index=row_labels, columns=["a", "b", "c", "d"])
        L_df = pd.DataFrame(L, index=row_labels, columns=["L"])
        pd.set_option("display.float_format", "{:>12.4f}".format)
        print("=" * 50)
        print("DESIGN MATRIX (A)")
        print("=" * 50)
        print(A_df.to_string())
        print("\n" + "=" * 50)
        print("OBSERVATION VECTOR (L)")
        print("=" * 50)
        print(L_df.to_string())

    # Least Squares Solution
    X = np.linalg.lstsq(A, L)[0]
    a, b, c, d = X.flatten()
    scale = np.sqrt(a**2 + b**2)
    theta = np.degrees(np.arctan2(b, a))

    if verbose:
        print("\n" + "=" * 50)
        print("CONFORMAL TRANSFORMATION PARAMETERS")
        print("=" * 50)
        print(f"a      = {a:.10f}")
        print(f"b      = {b:.10f}")
        print(f"c      = {c:.4f}")
        print(f"d      = {d:.4f}")
        print(f"Scale  = {scale:.10f}")
        print(f"Rotation = {theta:.6f}°")
        print("=" * 50)
    return a, b, c, d, scale, theta


# ============================================================
# 2. Transform Coordinates
# ============================================================
def transform_conformal(E, N, a, b, c, d):
    """
    Transform coordinates using conformal transformation.
    """
    E = np.asarray(E, dtype=float)
    N = np.asarray(N, dtype=float)
    E_new = a * E - b * N + c
    N_new = b * E + a * N + d
    print("\nTransformed Coordinates")
    print("-" * 50)
    print(f"Easting: {E_new}  Northing: {N_new}")
    return E_new, N_new


# ============================================================
# 3. Compute Residuals
# ============================================================
def residuals_conformal(E_s, N_s, E_T, N_T, a, b, c, d):
    """
    Compute residuals and RMSE.
    """
    E_calc, N_calc = transform_conformal(E_s, N_s, a, b, c, d)
    vE = np.asarray(E_T, dtype=float) - E_calc
    vN = np.asarray(N_T, dtype=float) - N_calc
    rmse_E = np.sqrt(np.mean(vE**2))
    rmse_N = np.sqrt(np.mean(vN**2))
    rmse_total = np.sqrt(np.mean(vE**2 + vN**2))
    print("\n")
    print("=" * 50)
    print("RESIDUALS")
    print("=" * 50)
    for i in range(len(vE)):
        print(
            f"Point {i+1}: "
            f"vE = {vE[i]:.4f} m, "
            f"vN = {vN[i]:.4f} m"
        )
    print("\nRMSE")
    print(f"Easting  : {rmse_E:.4f} m")
    print(f"Northing : {rmse_N:.4f} m")
    print(f"Overall  : {rmse_total:.4f} m")
    return vE, vN, rmse_total


# --------------------------
# AFFINE TRANSFORMATION
# --------------------------
def affine_transformation(E_s, N_s, E_T, N_T, E_new=None, N_new=None, verbose=True):
    """
    Computes a 2D affine transformation using least squares and
    optionally transforms new coordinates.

    Model:  E_T = a*E_s + b*N_s + c
            N_T = d*E_s + e*N_s + f
    """
    E_s = np.asarray(E_s, dtype=float)
    N_s = np.asarray(N_s, dtype=float)
    E_T = np.asarray(E_T, dtype=float)
    N_T = np.asarray(N_T, dtype=float)
    n = len(E_s)
    if not (len(N_s) == len(E_T) == len(N_T) == n):
        raise ValueError("All control point arrays must have the same length.")
    if n < 3:
        raise ValueError("At least 3 control points are required.")

    # Build Design Matrix
    A = np.zeros((2 * n, 6))
    L = np.zeros((2 * n, 1))
    for i in range(n):
        # Easting equation
        A[2*i] = [E_s[i], N_s[i], 1, 0, 0, 0]
        L[2*i] = E_T[i]
        # Northing equation
        A[2*i+1] = [0, 0, 0, E_s[i], N_s[i], 1]
        L[2*i+1] = N_T[i]

    if verbose:
        row_labels = []
        for i in range(n):
            row_labels += [f"E{i + 1}", f"N{i + 1}"]
        A_df = pd.DataFrame(A, index=row_labels,
                            columns=["a", "b", "c", "d", "e", "f"])
        L_df = pd.DataFrame(L, index=row_labels, columns=["L"])
        pd.set_option("display.float_format", "{:>12.4f}".format)
        print("=" * 45)
        print("DESIGN MATRIX (A)")
        print("=" * 45)
        print(A_df.to_string())
        print("\n" + "=" * 45)
        print("OBSERVATION VECTOR (L)")
        print("=" * 45)
        print(L_df.to_string())

    # Least Squares Solution
    X = np.linalg.lstsq(A, L)[0]
    a, b, c, d, e, f = X.flatten()

    if verbose:
        print("\n" + "=" * 45)
        print("Affine Transformation Parameters")
        print("=" * 45)
        print(f"a = {a:.10f}")
        print(f"b = {b:.10f}")
        print(f"c = {c:.4f}")
        print(f"d = {d:.10f}")
        print(f"e = {e:.10f}")
        print(f"f = {f:.4f}")
        print("=" * 45)

    transformed = None
    if E_new is not None and N_new is not None:
        E_new = np.atleast_1d(np.asarray(E_new, dtype=float))
        N_new = np.atleast_1d(np.asarray(N_new, dtype=float))
        E_trans = a * E_new + b * N_new + c
        N_trans = d * E_new + e * N_new + f
        transformed = (E_trans, N_trans)
        if verbose:
            print("\nTransformed Coordinates")
            print("-" * 45)
            for i in range(len(E_trans)):
                print(f"Point {i+1}:")
                print(f"E = {E_trans[i]:.4f}")
                print(f"N = {N_trans[i]:.4f}\n")
    return (a, b, c, d, e, f), transformed


# -------------------------------------------
# ECEF <=> GEODETIC
# -------------------------------------------
def geodetic_to_ecef(lat_str, lon_str, height):
    """
    Convert geodetic coordinates (latitude, longitude, height) to
    ECEF coordinates (X, Y, Z).  lat/lon as DMS strings, e.g. '43 15 46.289'.
    """
    a = 6378137.0
    f = 1 / 298.257223563
    ecc = math.sqrt(2 * f - f ** 2)

    lat_parts = lat_str.split()
    lat_deg = float(lat_parts[0])
    lat_min = float(lat_parts[1])
    lat_sec = float(lat_parts[2])
    lat_sign = -1.0 if lat_deg < 0 else 1.0
    lat_decimal = lat_sign * (abs(lat_deg) + lat_min / 60 + lat_sec / 3600)

    lon_parts = lon_str.split()
    lon_deg = float(lon_parts[0])
    lon_min = float(lon_parts[1])
    lon_sec = float(lon_parts[2])
    lon_sign = -1.0 if lon_deg < 0 else 1.0
    lon_decimal = lon_sign * (abs(lon_deg) + lon_min / 60 + lon_sec / 3600)

    lat_rad = math.radians(lat_decimal)
    lon_rad = math.radians(lon_decimal)

    N = a / math.sqrt(1 - (ecc ** 2) * (math.sin(lat_rad) ** 2))
    X = (N + height) * math.cos(lat_rad) * math.cos(lon_rad)
    Y = (N + height) * math.cos(lat_rad) * math.sin(lon_rad)
    Z = ((1 - ecc ** 2) * N + height) * math.sin(lat_rad)
    return X, Y, Z


def ecef_to_geodetic(X, Y, Z):
    """
    Convert ECEF coordinates (X, Y, Z) to geodetic coordinates
    (latitude in degrees, longitude in degrees, height in meters).
    """
    a = 6378137.0
    f = 1 / 298.257223563
    ecc = math.sqrt(2 * f - f ** 2)

    lon_rad = math.atan2(Y, X)
    p = math.sqrt(X ** 2 + Y ** 2)

    # initial approximation of latitude
    lat_rad = math.atan2(Z, p * (1 - ecc ** 2))

    # iterate to improve latitude
    for _ in range(5):
        N = a / math.sqrt(1 - (ecc ** 2) * (math.sin(lat_rad) ** 2))
        lat_rad = math.atan2(Z + ecc ** 2 * N * math.sin(lat_rad), p)

    # FIX: recompute N and height with the FINAL latitude so the
    # returned (lat, lon, height) triple is self-consistent.  Use the
    # Z/sin(phi) form near the poles where cos(phi) ~ 0.
    N = a / math.sqrt(1 - (ecc ** 2) * (math.sin(lat_rad) ** 2))
    if abs(math.cos(lat_rad)) > 1e-10:
        height = p / math.cos(lat_rad) - N
    else:
        height = Z / math.sin(lat_rad) - N * (1 - ecc ** 2)

    lat_deg = math.degrees(lat_rad)
    lon_deg = math.degrees(lon_rad)
    return lat_deg, lon_deg, height