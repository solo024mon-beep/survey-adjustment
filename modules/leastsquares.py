"""
leastsquares.py

Weighted least-squares adjustment

    delta       = (A^T P A)^-1 A^T P L        corrections to the unknowns
    V           = A.delta - L                 residuals
    sigma0^2    = (V^T P V) / (n - u)         a-posteriori variance of unit weight
    Qxx         = (A^T P A)^-1                cofactor matrix of the unknowns
    Cxx         = sigma0^2 . Qxx              covariance matrix of the unknowns
    std_dev     = sqrt(diag(Cxx))             standard deviations of the unknowns
    Qvv         = Qll - A . Qxx . A^T         cofactor matrix of the residuals
    sd_residual = sqrt(diag(sigma0^2 . Qvv))  standard deviations of the residuals

"""

import numpy as np


def least_squares(A: np.ndarray, L: np.ndarray, P: np.ndarray = None,
                   prov_values: np.ndarray = None, verbose: bool = True) -> dict:
    """
    Perform a weighted least-squares adjustment.

    Parameters
    ----------
    A : np.ndarray
        Design (coefficient) matrix, shape (n_obs, n_unknowns).
    L : np.ndarray
        Observation vector (obs - calc), shape (n_obs,) or (n_obs, 1).
    P : np.ndarray, optional
        Weight matrix, shape (n_obs, n_obs). Defaults to the identity
        matrix (every observation weighted equally) if omitted.
    prov_values : np.ndarray, optional
        Provisional values of the unknowns (e.g. provisional heights or
        coordinates), shape (n_unknowns,) or (n_unknowns, 1). If given,
        the function also returns final_values = prov_values + delta.
    verbose : bool, default True
        Print a full adjustment report. Set False to compute silently.

    Returns
    -------
    dict with keys:
        'delta'        : corrections to the unknowns, shape (n_unknowns, 1)
        'V'            : residuals, A.delta - L, shape (n_obs, 1)
        'N'            : normal matrix, A^T P A
        'U'            : normal equation vector, A^T P L
        'Qxx'          : cofactor matrix of unknowns, (A^T P A)^-1
        'sigma0_sq'    : a-posteriori variance factor, sigma0^2 (nan if dof <= 0)
        'sigma0'       : a-posteriori standard error, sigma0 (nan if dof <= 0)
        'Cxx'          : covariance matrix of unknowns, sigma0^2 . Qxx
        'std_dev'      : standard deviations of the unknowns, sqrt(diag(Cxx))
        'Qvv'          : cofactor matrix of the residuals, Qll - A.Qxx.A^T
        'sd_residual'  : standard deviations of the residuals, sqrt(diag(sigma0^2 . Qvv))
        'dof'          : degrees of freedom / redundancy, n_obs - n_unknowns
        'final_values' : prov_values + delta, or None if prov_values wasn't given
    """
    A = np.asarray(A, dtype=float)
    L = np.asarray(L, dtype=float).reshape(-1, 1)

    n_obs, n_unknowns = A.shape

    if L.shape[0] != n_obs:
        raise ValueError(f"L has {L.shape[0]} rows but A has {n_obs} rows -- they must match")

    if P is None:
        P = np.eye(n_obs)
    else:
        P = np.asarray(P, dtype=float)
        if P.shape != (n_obs, n_obs):
            raise ValueError(f"P must be shape ({n_obs}, {n_obs}), got {P.shape}")

    dof = n_obs - n_unknowns

    # ── Normal equations ────────────────────────────────────────────────
    N = A.T @ P @ A
    U = A.T @ P @ L

    try:
        Qxx = np.linalg.inv(N)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Normal matrix (A^T P A) is singular and can't be inverted. "
            "This usually means the design matrix has redundant/dependent "
            "columns -- e.g. an under-constrained network with too few "
            "control points, or duplicate/conflicting observation rows."
        ) from e

    delta = Qxx @ U
    V = A @ delta - L

    # ── A-posteriori variance of unit weight ───────────────────────────
    if dof > 0:
        sigma0_sq = float((V.T @ P @ V).item() / dof)
        sigma0 = np.sqrt(sigma0_sq)
    else:
        sigma0_sq = float('nan')
        sigma0 = float('nan')
        if verbose:
            print(f"\n⚠  Degrees of freedom = {dof} (n_obs={n_obs}, n_unknowns={n_unknowns}). "
                  f"There's no redundancy to estimate a variance of unit weight from, "
                  f"so sigma0^2 and the standard deviations below are undefined (nan).\n")

    Cxx = sigma0_sq * Qxx
    std_dev = np.sqrt(np.diag(Cxx))

    # ── Standard deviations of the residuals ────────────────────────────
    Qll = np.linalg.inv(P)
    Qvv = Qll - A @ Qxx @ A.T
    qvv_diag = np.diag(Qvv).copy()
    qvv_diag[np.isclose(qvv_diag, 0)] = 0.0  # guard tiny negative round-off
    sd_residual = np.sqrt(sigma0_sq * qvv_diag)

    # ── Optional: apply corrections to provisional values ──────────────
    final_values = None
    if prov_values is not None:
        prov_values = np.asarray(prov_values, dtype=float).reshape(-1, 1)
        if prov_values.shape[0] != n_unknowns:
            raise ValueError(f"prov_values must have {n_unknowns} entries "
                              f"(one per unknown), got {prov_values.shape[0]}")
        final_values = prov_values + delta

    if verbose:
        print("\n" + "=" * 70)
        print(" " * 20 + "LEAST SQUARES ADJUSTMENT")
        print("=" * 70)
        print(f"Observations (n) : {n_obs}")
        print(f"Unknowns     (u) : {n_unknowns}")
        print(f"Redundancy (dof) : {dof}")

        print(f"\nNormal matrix (N = A^T P A):\n{N.round(4)}")
        print(f"\nNormal equation vector (U = A^T P L):\n{U.round(4)}")
        print(f"\nCorrections to unknowns (delta):\n{delta.round(4)}")
        print(f"\nResiduals (V = A.delta - L):\n{V.round(4)}")
        print(f"\nA-posteriori variance of unit weight (sigma0^2): {sigma0_sq}")
        print(f"A-posteriori standard error (sigma0)            : {sigma0}")
        print(f"\nCofactor matrix of unknowns (Qxx):\n{Qxx.round(6)}")
        print(f"\nCovariance matrix of unknowns (Cxx = sigma0^2 . Qxx):\n{Cxx.round(6)}")
        print(f"\nStandard deviations of unknowns:\n{std_dev.round(4)}")
        print(f"\nStandard deviations of residuals:\n{sd_residual.round(4)}")

        if final_values is not None:
            print(f"\nFinal adjusted values (provisional + delta):\n{final_values.round(4)}")

        print("=" * 70 + "\n")

    return {
        'delta': delta,
        'V': V,
        'N': N,
        'U': U,
        'Qxx': Qxx,
        'sigma0_sq': sigma0_sq,
        'sigma0': sigma0,
        'Cxx': Cxx,
        'std_dev': std_dev,
        'Qvv': Qvv,
        'sd_residual': sd_residual,
        'dof': dof,
        'final_values': final_values,
    }
