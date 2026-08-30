"""
leveling_adjustment.py
Least-squares adjustment of a differential leveling network.

Given a list of leveling runs (from, to, observed elevation difference,
and either the run's length or its standard deviation), this builds and
solves the standard indirect (parametric) least-squares model:

    L + v = A x        (equivalently: v = A x - L)
    x    = (A' P A)^-1 A' P L         adjusted heights (unknowns)
    v    = A x - L                    residuals
    r    = m - u                      redundancy (obs - unknowns)
    s0^2 = (v' P v) / r               a-posteriori variance of unit weight
    Qxx  = (A' P A)^-1                cofactor matrix of the unknowns
    Qll  = P^-1                       cofactor matrix of the observations
    Qvv  = Qll - A Qxx A'             cofactor matrix of the residuals

Sign/model convention
---------------------
Unknown x_i is the elevation of point i. For an observed run from F to T
with observed elevation difference dh (T minus F, as normally recorded
in a leveling field book):

    dh_obs (+v) = H_T - H_F

If both F and T are unknown (not control):  A row has -1 at F, +1 at T,
L = dh_obs.
If F is a fixed control point (known height H_F) and T is unknown:
A row has +1 at T only, L = dh_obs + H_F.
If T is a fixed control point and F is unknown:
A row has +1 at F only, L = H_T - dh_obs.
If both F and T are control points, the run carries no information
about any unknown -- it is reported separately as a control-to-control
check (misclosure) and excluded from the adjustment itself.

Weights
-------
If `length` is given:  P_ii = c / length_i   (c defaults to 1; use the
shortest run's length, or any convenient constant -- only the relative
weights matter).
If `std` is given directly: P_ii = 1 / std_i^2.

Rejection criteria (blunder detection)
--------------------------------------
Baarda's data-snooping w-test is used: for each observation,

    w_i = v_i / (s0 * sqrt(Qvv_ii))

is compared against the chosen critical multiplier `multiplier` (default 
0.6745 for 50% confidence, but you can set it to 1.6449 for 90%, 1.9600 
for 95%, 2.5758 for 99%, or any other value). Observations with |w_i| 
above the multiplier are flagged as likely blunders.
A global model test (chi-square test on s0^2 against sigma0_apriori^2, 
df = r) is also reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------
# Core adjustment
# ---------------------------------------------------------------------
def leveling_adjustment(
    obs: pd.DataFrame,
    control: dict,
    from_col: str = "from",
    to_col: str = "to",
    dh_col: str = "dh",
    length_col: str | None = "length",
    std_col: str | None = None,
    c: float = 1.0,
    sigma0_apriori: float = 1.0,
    alpha: float = 0.05,
    multiplier: float | None = None,  # NEW: manual multiplier override
    provisional_heights: dict | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run a full least-squares adjustment of a differential leveling network.

    Two equivalent ways to set up the model -- pick one with
    `provisional_heights`:

    1. Direct model (default, `provisional_heights=None`): unknowns ARE
       the final adjusted heights. L is built by moving each row's known
       control value(s) to the other side of the equation. No provisional
       estimate is needed for the unknown points.
    2. Provisional/correction model (`provisional_heights` given):
       unknowns are CORRECTIONS to a supplied approximate height for each
       unknown point (final height = provisional + correction). L is the
       misclosure for every row, written in the field-book/lecture layout
       (a row containing a single unknown always carries +1 on that
       unknown; rows that end on a control point are multiplied by -1
       relative to the uniform form, which is mathematically identical):

           L_i = dh_obs,i - (H_T,provisional - H_F,provisional)

       using the exact known value for control points and your supplied
       estimate for unknown points. This is the standard "provisional
       heights" formulation taught in many surveying courses, and gives
       identical final adjusted heights/residuals/sigma0 to the direct
       model -- it's just a different parameterization of the same
       problem.

    Parameters
    ----------
    obs : pd.DataFrame
        One row per leveling run, with columns `from_col`, `to_col`,
        `dh_col`, and EITHER `length_col` OR `std_col` (whichever is not
        None / not present is ignored). `dh` is the observed elevation
        change from the 'from' point to the 'to' point (to - from).
    control : dict
        Mapping {point_label: known_elevation} for fixed/control points.
        These are removed from the list of unknowns; every other point
        that appears in `obs` becomes an unknown to be adjusted.
    from_col, to_col, dh_col : str
        Column names in `obs`.
    length_col : str or None
        Column name holding run lengths (any consistent unit). Used for
        weighting as P = c / length if present and `std_col` is None.
    std_col : str or None
        Column name holding each run's standard deviation. If given
        (and not None), this takes priority over `length_col` for
        weighting: P = 1 / std^2.
    c : float, default 1.0
        Weighting constant used with `length_col` (P_ii = c / length_i).
    sigma0_apriori : float, default 1.0
        A-priori standard deviation of unit weight, used only for the
        global chi-square model test.
    alpha : float, default 0.05
        Significance level for the global chi-square test. 
        NOTE: For the rejection criterion (Baarda w-test), use `multiplier`
        to manually set the critical value instead of computing from alpha.
    multiplier : float, optional
        Critical multiplier for the Baarda w-test. If provided, this value
        is used directly instead of computing from alpha using the normal
        distribution. Common values:
            - 0.6745  for 50% confidence (E50)
            - 1.6449  for 90% confidence (E90)
            - 1.9600  for 95% confidence
            - 2.5758  for 99% confidence (E99)
        If None (default), the value is automatically computed from
        the normal distribution using `alpha`.
    provisional_heights : dict, optional
        Mapping {point_label: approximate_elevation} for every unknown
        point (control points don't need an entry -- their exact known
        value is used automatically). If given, switches to the
        provisional/correction model described above.
    verbose : bool, default True
        Print a full formatted report.

    Returns
    -------
    dict with keys:
        'A'                 design matrix (DataFrame, obs x unknowns)
        'L'                 observation vector (Series)
        'P'                 weight matrix (DataFrame, diagonal)
        'unknowns'          list of unknown point labels, in x/Qxx order
        'x'                 solved unknowns: adjusted heights (direct
                             model) or corrections (provisional model) --
                             see 'adjusted_heights' for final heights
                             either way
        'adjusted_heights'  adjusted heights for ALL points, unknown +
                             control (Series)
        'v'                 residuals (Series, indexed like obs)
        'sigma0'            a-posteriori standard deviation of unit weight
        'Qxx'               cofactor matrix of unknowns (DataFrame)
        'Qll'               cofactor matrix of observations (DataFrame)
        'Qvv'               cofactor matrix of residuals (DataFrame)
        'var_heights'       variances of adjusted heights (Series)
        'std_heights'       standard errors of adjusted heights (Series)
        'std_residuals'     standard errors of residuals (Series)
        'standardized_resid' Baarda w-statistic per observation (Series)
        'flagged'           DataFrame of observations flagged as
                             possible blunders (subset of `obs`, with
                             v, std_residual, w columns added)
        'redundancy'        r = m - u
        'chi2_stat', 'chi2_crit_low', 'chi2_crit_high', 'chi2_pass'
                            global model test results
        'control_checks'    DataFrame of control-to-control runs
                             (excluded from the adjustment) with their
                             raw misclosure vs. known control values
        'obs_used'          the subset of `obs` actually used in the
                             adjustment (control-to-control rows removed)
        'multiplier'        the critical multiplier used for rejection
    """
    obs = obs.copy().reset_index(drop=True)

    # ---- figure out unknowns (every point in obs not in `control`) ---- #
    all_points = pd.unique(obs[[from_col, to_col]].values.ravel())
    unknowns = [p for p in all_points if p not in control]
    unknowns.sort()
    u_index = {p: j for j, p in enumerate(unknowns)}
    u = len(unknowns)
    if u == 0:
        raise ValueError("No unknown points found -- every point in `obs` "
                         "is already in `control`. Nothing to adjust.")

    # ---- separate out control-to-control runs (no unknowns involved) -- #
    is_ctrl_to_ctrl = obs[from_col].isin(control) & obs[to_col].isin(control)
    control_checks = obs.loc[is_ctrl_to_ctrl].copy()
    if len(control_checks):
        control_checks["misclosure"] = (
            control_checks.apply(
                lambda r: (control[r[to_col]] - control[r[from_col]])
                - r[dh_col], axis=1)
        )
    obs_used = obs.loc[~is_ctrl_to_ctrl].copy().reset_index(drop=True)
    if len(obs_used) == 0:
        raise ValueError("Every observation is between two control points "
                         "-- there is nothing left to adjust.")
    m = len(obs_used)

    # ------------------------------- weights ---------------------------- #
    if std_col is not None and std_col in obs_used.columns:
        stds = obs_used[std_col].astype(float).to_numpy()
        if np.any(stds <= 0):
            raise ValueError("std_col contains zero/negative standard "
                             "deviations -- weights would be undefined.")
        weights = 1.0 / stds ** 2
        weight_source = f"std ('{std_col}')"
    elif length_col is not None and length_col in obs_used.columns:
        lengths = obs_used[length_col].astype(float).to_numpy()
        if np.any(lengths <= 0):
            raise ValueError("length_col contains zero/negative lengths -- "
                             "weights would be undefined.")
        weights = c / lengths
        weight_source = f"length ('{length_col}'), c={c}"
    else:
        raise ValueError("Need either a valid `std_col` or `length_col` "
                         "present in `obs` to compute weights.")
    P = np.diag(weights)

    # ---- validate provisional_heights, if given ---- #
    use_provisional = provisional_heights is not None
    if use_provisional:
        missing = [p for p in unknowns if p not in provisional_heights]
        if missing:
            raise ValueError(f"provisional_heights is missing an estimate "
                             f"for unknown point(s): {missing}. Every "
                             f"unknown point needs a provisional/"
                             f"approximate elevation in this mode.")

        def h_prov(pt):
            return control[pt] if pt in control else provisional_heights[pt]

    # --------------------------- design matrix --------------------------- #
    A = np.zeros((m, u))
    L = np.zeros(m)
    for i, row in obs_used.iterrows():
        f, t, dh = row[from_col], row[to_col], float(row[dh_col])
        f_unknown = f in u_index
        t_unknown = t in u_index
        if use_provisional:
            if f_unknown and t_unknown:
                # unknown -> unknown: x_to - x_from = dh - (prov_to - prov_from)
                A[i, u_index[f]] = -1.0
                A[i, u_index[t]] = 1.0
                L[i] = dh - (h_prov(t) - h_prov(f))
            elif f_unknown:
                # unknown -> CONTROL: write the single unknown with +1,
                # x_from = (H_control - prov_from) - dh   (row x -1 vs the
                # uniform form -- identical equation, matches the
                # field-book/lecture layout)
                A[i, u_index[f]] = 1.0
                L[i] = (control[t] - h_prov(f)) - dh
            else:
                # CONTROL -> unknown:  x_to = dh + H_control - prov_to
                A[i, u_index[t]] = 1.0
                L[i] = dh - (h_prov(t) - control[f])
        else:
            # Direct model: unknowns ARE the final heights, so a control
            # endpoint's exact value gets moved to the other side of the
            # equation instead of getting a column.
            if f_unknown and t_unknown:
                A[i, u_index[f]] = -1.0
                A[i, u_index[t]] = 1.0
                L[i] = dh
            elif f_unknown and not t_unknown:      # T is control
                A[i, u_index[f]] = 1.0
                L[i] = control[t] - dh
            elif not f_unknown and t_unknown:      # F is control
                A[i, u_index[t]] = 1.0
                L[i] = dh + control[f]
            else:
                # Shouldn't happen -- these rows were filtered into
                # control_checks above.
                raise RuntimeError(f"Unexpected control-control row at "
                                   f"index {i}")

    # ------------------------------ solve --------------------------------- #
    N = A.T @ P @ A
    try:
        Qxx = np.linalg.inv(N)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(
            "Normal matrix A'PA is singular -- the network is not fully "
            "connected to the control (some unknown point(s) may have no "
            "path of observations back to a fixed point), or there are "
            "duplicate/redundant unknown labels. Original error: "
            f"{e}"
        )
    t_vec = A.T @ P @ L
    x = Qxx @ t_vec
    v = A @ x - L
    r = m - u
    if r < 0:
        raise ValueError(f"Network is under-determined: {m} observations "
                         f"but {u} unknowns (redundancy = {r} < 0). Add "
                         f"more observations or fix more points as "
                         f"control.")
    if r == 0:
        sigma0_hat = np.nan
        Qvv = np.zeros((m, m))
        std_residuals = np.full(m, np.nan)
        standardized = np.full(m, np.nan)
    else:
        sigma0_hat_sq = (v @ P @ v) / r
        sigma0_hat = np.sqrt(sigma0_hat_sq)
        Qll = np.linalg.inv(P)
        Qvv = Qll - A @ Qxx @ A.T
        diag_qvv = np.clip(np.diag(Qvv), 0, None)  # guard tiny negative noise
        std_residuals = sigma0_hat * np.sqrt(diag_qvv)
        with np.errstate(invalid="ignore", divide="ignore"):
            standardized = np.where(std_residuals > 0,
                                    v / np.where(std_residuals > 0,
                                                 std_residuals, np.nan),
                                    np.nan)
    var_heights = (sigma0_hat ** 2 if r > 0 else sigma0_apriori ** 2) * np.diag(Qxx)
    std_heights = np.sqrt(var_heights)

    # ------------------------ rejection criteria --------------------------- #
    # Use manual multiplier if provided, otherwise compute from alpha
    if multiplier is not None:
        crit = multiplier
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
    
    if r > 0:
        flagged_mask = np.abs(standardized) > crit
    else:
        flagged_mask = np.zeros(m, dtype=bool)
    flagged = obs_used.loc[flagged_mask].copy()
    flagged["v"] = v[flagged_mask]
    flagged["std_residual"] = std_residuals[flagged_mask]
    flagged["w"] = standardized[flagged_mask]

    # ------------------------- global chi-square test ------------------------ #
    if r > 0:
        chi2_stat = r * (sigma0_hat ** 2) / (sigma0_apriori ** 2)
        chi2_lo = stats.chi2.ppf(alpha / 2, r)
        chi2_hi = stats.chi2.ppf(1 - alpha / 2, r)
        chi2_pass = bool(chi2_lo <= chi2_stat <= chi2_hi)
    else:
        chi2_stat = chi2_lo = chi2_hi = np.nan
        chi2_pass = None

    # ------------------------------- package up ------------------------------ #
    x_series = pd.Series(x, index=unknowns, name="x")
    if use_provisional:
        # x holds CORRECTIONS in this mode -- final height = provisional + x
        prov_series = pd.Series({p: provisional_heights[p] for p in unknowns})
        unk_series = (prov_series + x_series).rename("adjusted_height")
    else:
        unk_series = x_series.rename("adjusted_height")
    ctrl_series = pd.Series(control, name="adjusted_height")
    adjusted_heights = pd.concat([unk_series, ctrl_series]).sort_index()
    A_df = pd.DataFrame(A, index=obs_used.index, columns=unknowns)
    L_series = pd.Series(L, index=obs_used.index, name="L")
    P_df = pd.DataFrame(P, index=obs_used.index, columns=obs_used.index)
    Qxx_df = pd.DataFrame(Qxx, index=unknowns, columns=unknowns)
    Qll_df = pd.DataFrame(np.linalg.inv(P), index=obs_used.index,
                          columns=obs_used.index)
    Qvv_df = pd.DataFrame(Qvv, index=obs_used.index, columns=obs_used.index)
    v_series = pd.Series(v, index=obs_used.index, name="residual")
    std_res_series = pd.Series(std_residuals, index=obs_used.index,
                               name="std_residual")
    w_series = pd.Series(standardized, index=obs_used.index, name="w")
    var_heights_series = pd.Series(var_heights, index=unknowns,
                                   name="variance")
    std_heights_series = pd.Series(std_heights, index=unknowns,
                                   name="std_error")
    result = {
        "A": A_df,
        "L": L_series,
        "P": P_df,
        "unknowns": unknowns,
        "x": x_series,
        "adjusted_heights": adjusted_heights,
        "v": v_series,
        "sigma0": sigma0_hat,
        "sigma0_apriori": sigma0_apriori,
        "Qxx": Qxx_df,
        "Qll": Qll_df,
        "Qvv": Qvv_df,
        "var_heights": var_heights_series,
        "std_heights": std_heights_series,
        "std_residuals": std_res_series,
        "standardized_resid": w_series,
        "flagged": flagged,
        "reject_critical_value": crit,
        "alpha": alpha,
        "multiplier": crit,  # store the actual multiplier used
        "redundancy": r,
        "chi2_stat": chi2_stat,
        "chi2_crit_low": chi2_lo,
        "chi2_crit_high": chi2_hi,
        "chi2_pass": chi2_pass,
        "control_checks": control_checks,
        "obs_used": obs_used,
        "weight_source": weight_source,
        "control": control,
        "use_provisional": use_provisional,
    }
    if verbose:
        _print_report(result, from_col, to_col, dh_col)
    return result


# ---------------------------------------------------------------------
# Elevation difference between any two stations, with propagated error
# ---------------------------------------------------------------------
def elevation_difference(result: dict, point1: str, point2: str) -> dict:
    """
    Estimate the adjusted elevation difference (point2 - point1) between
    any two stations in the network (control or adjusted-unknown), with
    its propagated standard error.

    Uses law of propagation of variance:

         Var(H2 - H1) = sigma0^2 * (Qxx[1,1] + Qxx[2,2] - 2*Qxx[1,2])

    Control points contribute zero variance/covariance (treated as exact).

    Returns
    -------
    dict with 'diff', 'std_error', 'point1', 'point2'.
    """
    heights = result["adjusted_heights"]
    if point1 not in heights.index or point2 not in heights.index:
        missing = [p for p in (point1, point2) if p not in heights.index]
        raise KeyError(f"Point(s) not found in adjusted network: {missing}")
    diff = heights[point2] - heights[point1]
    Qxx = result["Qxx"]
    sigma0 = result["sigma0"]
    unknowns = result["unknowns"]
    q11 = Qxx.loc[point1, point1] if point1 in unknowns else 0.0
    q22 = Qxx.loc[point2, point2] if point2 in unknowns else 0.0
    q12 = (Qxx.loc[point1, point2]
           if (point1 in unknowns and point2 in unknowns) else 0.0)
    var = (sigma0 ** 2) * (q11 + q22 - 2 * q12) if not np.isnan(sigma0) else np.nan
    std_err = np.sqrt(var) if var is not np.nan and var >= 0 else np.nan
    return {"point1": point1, "point2": point2, "diff": diff,
            "std_error": std_err}


# ---------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------
def _print_report(result: dict, from_col: str, to_col: str, dh_col: str) -> None:
    pd.set_option("display.float_format", "{:>10.5f}".format)
    pd.set_option("display.width", 140)
    print("\n" + "=" * 90)
    print(" " * 30 + "LEVELING NETWORK ADJUSTMENT")
    print("=" * 90)
    print(f"\nControl (fixed) points: {result['control']}")
    print(f"Unknown points ({len(result['unknowns'])}): {result['unknowns']}")
    print(f"Weighting based on: {result['weight_source']}")
    model_desc = ("provisional/correction (x = corrections to your supplied "
                  "estimates)" if result["use_provisional"]
                  else "direct (x = final adjusted heights)")
    print(f"Model: {model_desc}")
    if len(result["control_checks"]):
        print("\n--- Control-to-control runs (excluded from adjustment) ---")
        print(result["control_checks"].to_string(index=False))
    print("\n--- Design matrix A ---")
    print(result["A"].to_string())
    print("\n--- Observation vector L ---")
    print(result["L"].to_string())
    print("\n--- Weight matrix P (diagonal shown) ---")
    print(pd.Series(np.diag(result["P"]), index=result["P"].index,
                    name="weight").to_string())
    print("\n--- Adjusted heights (all points) ---")
    print(result["adjusted_heights"].to_string())
    print(f"\n--- Unknowns: solved x ({'corrections' if result['use_provisional'] else 'final heights'}), "
          f"final adjusted height, variance, std. error ---")
    summary = pd.DataFrame({
        "x": result["x"],
        "adjusted_height": result["adjusted_heights"].loc[result["unknowns"]],
        "variance": result["var_heights"],
        "std_error": result["std_heights"],
    })
    print(summary.to_string())
    print(f"\nRedundancy (r = m - u): {result['redundancy']}")
    print(f"\nA-posteriori sigma0: {result['sigma0']:.5f}"
          if not np.isnan(result["sigma0"]) else
          "A-posteriori sigma0: undefined (r = 0, no redundancy)")
    print("\n--- Residuals & standard errors ---")
    resid_summary = pd.DataFrame({
        from_col: result["obs_used"][from_col].values,
        to_col: result["obs_used"][to_col].values,
        dh_col: result["obs_used"][dh_col].values,
        "residual": result["v"].values,
        "std_residual": result["std_residuals"].values,
        "w (standardized)": result["standardized_resid"].values,
    })
    print(resid_summary.to_string(index=False))
    print(f"\n--- Rejection criteria check (Baarda w-test) ---")
    print(f"Critical multiplier: {result['multiplier']:.4f}")
    if len(result["flagged"]):
        print("Flagged observations (possible blunders):")
        print(result["flagged"].to_string(index=False))
    else:
        print("No observations flagged.")
    print(f"\n--- Global model test (chi-square, alpha={result['alpha']}) ---")
    if result["chi2_pass"] is None:
        print("Not computable (r = 0, no redundancy).")
    else:
        print(f"chi2_stat = {result['chi2_stat']:.4f}, acceptance range = "
              f"[{result['chi2_crit_low']:.4f}, {result['chi2_crit_high']:.4f}]")
        print("Result: PASS (sigma0 not significantly different from "
              "a-priori)" if result["chi2_pass"] else
              "Result: FAIL (check for undetected blunders or "
              "mis-specified weights)")
    print("=" * 90 + "\n")