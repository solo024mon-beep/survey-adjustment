"""
residual_test.py

Individual observation testing (data snooping) that follows a weighted
least-squares adjustment. Implements the workflow:

    Step 1  Observations              L
    Step 2  Design matrix             A
    Step 3  Weight matrix             P
    Step 4  Least-squares adjustment  x_hat = (A'PA)^-1 A'PL
    Step 5  Residuals                 v = A.x_hat - L
    Step 6  Residual cofactors        Qvv = Qll - A Qxx A'      (Qll = P^-1)
    Step 7  Test statistic per obs.   w_i (or tau_i)
    Step 8  Critical value            Rc
    Step 9  Decision                  |test value| > Rc -> REJECTED, else ACCEPTED

Two test types are supported:

  'w'   -- Baarda's w-test. Assumes the reference variance sigma0 is
           KNOWN a priori (default sigma0 = 1.0, i.e. your weights are
           already true inverse-variances). Test statistic:
               w_i = v_i / (sigma0 * sqrt(Qvv_ii))
           Critical value from the standard normal distribution.

  'tau' -- Pope's tau-test. Use this when sigma0 is NOT known a priori
           and must be estimated from the adjustment itself (the usual
           case in practice). Test statistic:
               tau_i = v_i / (sigma0_hat * sqrt(Qvv_ii))
           Critical value from the Student's t-distribution with
           (dof - 1) degrees of freedom, which properly accounts for
           sigma0_hat being an estimate rather than a known constant.

If you don't know which one your course/workflow expects: 'w' is what
the simple "Value vs Rc" diagram usually means when sigma0 is just
assumed to be 1; 'tau' is the more defensible choice whenever sigma0
is being estimated from the same adjustment (which is the normal case
for survey network adjustments).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t


def test_residuals(A, L=None, P=None, row_labels=None, alpha: float = 0.05,
                    confidence: float = None, sigma0: float = 1.0,
                    test_type: str = 'w', verbose: bool = True,
                    least_squares_result: dict = None,
                    multiplier: float = None) -> pd.DataFrame:  # NEW: manual multiplier
    """
    Test every observation's residual against a critical value, following
    a weighted least-squares adjustment.

    Two ways to call this:

      1. Pass A, L, P -- the adjustment is computed from scratch here
         (steps 4-9 of the workflow, all in this function).

      2. Pass A, P, and least_squares_result=<the dict returned by
         leastsquares.least_squares(...)> -- REUSES the residuals (V),
         cofactor matrix (Qxx), dof, and sigma0_hat that were already
         computed there instead of re-solving the adjustment. This is
         the preferred way to call it if you've already run
         least_squares() -- it avoids duplicate work and guarantees the
         two functions agree on the same solution.

    Parameters
    ----------
    A : array-like
        Design (coefficient) matrix, shape (n_obs, n_unknowns).
    L : array-like, optional
        Observation vector (obs - calc), shape (n_obs,) or (n_obs, 1).
        Required only if least_squares_result is not given.
    P : array-like, optional
        Weight matrix, shape (n_obs, n_obs). Defaults to the identity
        matrix if omitted.
    row_labels : list[str], optional
        Label for each observation row (e.g. ['Dist_AB', 'Bear_AB', ...]).
        Defaults to 'Obs_1', 'Obs_2', ... if omitted.
    alpha : float, default 0.05
        Two-tailed significance level (e.g. 0.05 -> 95% confidence).
        Ignored if `confidence` is given.
        NOTE: For the rejection criterion, use `multiplier` to manually
        set the critical value instead of computing from alpha.
    confidence : float, optional
        Confidence level as an alternative to alpha -- e.g. an exam
        question asking to "check for outliers at 50% confidence" is
        confidence=0.50 (or confidence=50, percentage form is also
        accepted). Internally this sets alpha = 1 - confidence. Matches
        the same E50/E90/E99 convention used in survey_stats.py:
        confidence=0.50 -> Rc=0.6745, confidence=0.90 -> Rc=1.6449,
        confidence=0.99 -> Rc=2.5758.
    sigma0 : float, default 1.0
        A-priori reference standard deviation of unit weight. Only used
        directly when test_type='w'; for test_type='tau' it's replaced
        by the a-posteriori sigma0_hat from the adjustment.
    test_type : {'w', 'tau'}, default 'w'
        Which test to run -- see module docstring.
    verbose : bool, default True
        Print the formatted results table. Set False to build silently.
    least_squares_result : dict, optional
        The dict returned by leastsquares.least_squares(A, L, P). If
        given, V/Qxx/dof/sigma0_hat are taken from it directly and L
        does not need to be passed again.
    multiplier : float, optional
        Critical multiplier for the test. If provided, this value is
        used directly as Rc instead of computing from alpha using the
        normal or t-distribution. Common values:
            - 0.6745  for 50% confidence (E50)
            - 1.6449  for 90% confidence (E90)
            - 1.9600  for 95% confidence
            - 2.5758  for 99% confidence (E99)
        If None (default), the value is automatically computed from
        alpha using the appropriate distribution (normal for 'w',
        t-distribution for 'tau').

    Returns
    -------
    pd.DataFrame
        Columns: Line, Residual (v), Qvv, Test Value, Critical Value (Rc), Decision
    """
    if confidence is not None:
        if confidence > 1:          # someone passed e.g. 50 instead of 0.50
            confidence = confidence / 100
        if not (0 < confidence < 1):
            raise ValueError(f"confidence must be between 0 and 1 (or 0 and 100 "
                              f"as a percentage), got {confidence}")
        alpha = 1 - confidence

    A = np.asarray(A, dtype=float)
    n_obs, n_unknowns = A.shape

    if P is None:
        P = np.eye(n_obs)
    else:
        P = np.asarray(P, dtype=float)
        if P.shape != (n_obs, n_obs):
            raise ValueError(f"P must be shape ({n_obs}, {n_obs}), got {P.shape}")

    if row_labels is None:
        row_labels = [f'Obs_{i + 1}' for i in range(n_obs)]
    if len(row_labels) != n_obs:
        raise ValueError(f"row_labels must have {n_obs} entries, got {len(row_labels)}")

    if least_squares_result is not None:
        # ── Reuse an already-computed adjustment ────────────────────────
        V = np.asarray(least_squares_result['V'], dtype=float).reshape(-1, 1)
        Qxx = np.asarray(least_squares_result['Qxx'], dtype=float)
        dof = least_squares_result['dof']
        sigma0_hat = least_squares_result['sigma0']
        if V.shape[0] != n_obs:
            raise ValueError(f"least_squares_result's V has {V.shape[0]} rows "
                              f"but A has {n_obs} rows -- did these come from "
                              f"the same adjustment?")
    else:
        # ── Compute the adjustment here (steps 4-5) ─────────────────────
        if L is None:
            raise ValueError("Pass either L (to compute the adjustment here) "
                              "or least_squares_result (to reuse one already "
                              "computed by leastsquares.least_squares).")
        L = np.asarray(L, dtype=float).reshape(-1, 1)

        dof = n_obs - n_unknowns
        if dof <= 0:
            raise ValueError(f"No redundancy to test (dof={dof}); need more "
                              f"observations than unknowns.")

        N = A.T @ P @ A
        try:
            Qxx = np.linalg.inv(N)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(
                "Normal matrix (A'PA) is singular -- check for redundant "
                "columns or an under-constrained network."
            ) from e

        U = A.T @ P @ L
        x_hat = Qxx @ U
        V = A @ x_hat - L

        sigma0_hat_sq = float((V.T @ P @ V).item() / dof)
        sigma0_hat = np.sqrt(sigma0_hat_sq)

    if dof <= 0:
        raise ValueError(f"No redundancy to test (dof={dof}); need more "
                          f"observations than unknowns.")

    # ── Step 6: residual cofactors ──────────────────────────────────────
    Qll = np.linalg.inv(P)
    Qvv = Qll - A @ Qxx @ A.T
    qvv_diag = np.diag(Qvv).copy()

    # Guard against tiny negative values from floating-point round-off
    qvv_diag[np.isclose(qvv_diag, 0)] = 0.0
    if np.any(qvv_diag < 0):
        raise ValueError("Qvv has negative diagonal entries -- check A/P for consistency.")

    # ── Step 7: standard deviation of each residual (Sd-Residual) ────────
    # This is sigma * sqrt(Qvv_ii) -- the ACTUAL standard deviation of the
    # residual (matching leastsquares.py's 'sd_residual'), not just the
    # raw cofactor sqrt(Qvv_ii). Using sigma0 for the w-test (the assumed
    # reference std dev) and sigma0_hat for the tau-test (the estimated
    # one) means Value = Residual / Sd-Residual holds exactly in the
    # table below, and -- when test_type='tau' and least_squares_result
    # is passed in -- this column matches leastsquares.py's sd_residual
    # for the same observations exactly, since both are then scaled by
    # the same sigma0_hat.
    with np.errstate(divide='ignore', invalid='ignore'):
        qvv_sqrt = np.sqrt(qvv_diag)

    # ── Determine the critical multiplier ──────────────────────────────
    # Use manual multiplier if provided, otherwise compute from alpha
    if multiplier is not None:
        z_or_t = multiplier
    else:
        if test_type == 'w':
            z_or_t = norm.ppf(1 - alpha / 2)
        elif test_type == 'tau':
            z_or_t = student_t.ppf(1 - alpha / 2, dof - 1) if dof > 1 else np.nan
        else:
            raise ValueError("test_type must be 'w' or 'tau'")

    # ── Step 7 (cont'd): Value = Residual / Sd-Residual ──────────────────
    if test_type == 'w':
        sd_residual = sigma0 * qvv_sqrt
    elif test_type == 'tau':
        sd_residual = sigma0_hat * qvv_sqrt

    with np.errstate(divide='ignore', invalid='ignore'):
        value = V.flatten() / sd_residual

    # ── Step 8: Rc = multiplier value ────────────────────────────────────
    Rc = np.full(n_obs, z_or_t)

    # ── Step 9: decision ─────────────────────────────────────────────────
    with np.errstate(invalid='ignore'):
        decision = np.where(np.abs(value) > Rc, 'Rejected', 'Accepted')

    df = pd.DataFrame({
        'Line': row_labels,
        'Residual': V.flatten(),
        'Sd-Residual': sd_residual,
        'Value': value,
        'Rc': Rc,
        'Accepted/Rejected': decision,
    })

    if verbose:
        pd.set_option('display.float_format', '{:>10.5f}'.format)
        pd.set_option('display.width', 130)
        title = f"INDIVIDUAL OBSERVATION TEST -- {test_type}-test (alpha={alpha})"
        print("\n" + "=" * 100)
        print(" " * ((100 - len(title)) // 2) + title)
        print("=" * 100)
        print(df.to_string(index=False))
        if test_type == 'tau':
            print(f"\nsigma0_hat (a-posteriori) = {sigma0_hat:.5f}")
        print(f"Critical z/t value used: {z_or_t:.5f}")
        print("=" * 100 + "\n")

    return df