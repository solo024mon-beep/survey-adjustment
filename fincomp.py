import pandas as pd
from modules.errorellipse import error_ellipse
from modules.Survey import traverse_tables
from modules.control_point_reducer import (interactive_reduce, build_prov_values)
from modules.leastsquares import least_squares
from modules.residual_test import test_residuals
from modules.survey_stats import correlation
from modules.leveling_adjustment import (
    leveling_adjustment, 
    elevation_difference
    )
from modules.transformation import (
    solve_conformal,
    transform_conformal,
    residuals_conformal,
)


# ============================================================
# TRAVERSING
# ============================================================

Lines = ['AB', 'BC', 'CD','DE','EA','BE','BF','FE','FC','FD']

obs_bearing = [
    '30 29 59.43','275 52 44.57','157 9 31.46',
    '254 11 36.25','292 58 21.82','163 38 3.41',
    '135 50 0','205 59 38.4','45 51 35.8','105 26 12.94']

cal_bearing = ['30 29 59.43','275 52 44.57','157 9 31.46',
               '254 11 36.25','292 58 21.82','163 38 3.41',
               '135 50 0','205 59 38.4','45 51 35.8','105 26 12.94']

obs_dist = [
    3669.240,4397.254,3101.625,
    4420.055,3462.076,4703.319,
    3369.030,2332.063,2823.857,
    3351.737
]

cal_dist = [
    3669.257045,4397.319396,3101.597309,
    4420.064519,3462.107195,4703.346776,
    3368.991824,2332.059244,2823.854488,
    3351.756805
]

dist_std = [0.013,0.015,0.012,0.015,0.013,0.016,0.012,0.010,0.011,0.012]
bearing_std = [0.4,1.3,2.1,2.5,2.5,2.5,1.1,1.2,1.2,2.5]


# Build the traverse adjustment table
traverse_df = traverse_tables(
    Lines,
    obs_bearing,
    cal_bearing,
    obs_dist,
    cal_dist
)


# Build the design matrix (A), observation vector (L), and weight matrix (P) for the traverse adjustment
# and filtered to whichever observation type(s) you pick at the
# prompt. dist_std/bearing_std are the per-line std devs already
# defined above, so interactive_reduce() uses them directly
A_reduced, L_reduced, P, unknown_stations, kept_labels, row_labels = (
    interactive_reduce(
        Lines,
        obs_bearing,
        cal_bearing,
        obs_dist,
        cal_dist,
        dist_std=dist_std,
        bearing_std=bearing_std,
    )
)


# Provisional coordinates for the unknown stations
# Must cover every station in `unknown_stations`.
prov_coords: dict[str, tuple[float, float]] = {
    'B': (12349.5,14708.750),
    'D': (17927.677,11399.956),
    'E': (13674.750,10195.970),
    'F': (14696.838,12292.118)
}

prov_values: list[float] = build_prov_values(
    unknown_stations,
    prov_coords
) # type: ignore[assignment,arg-type]


# Least-squares computation for the survey
result = least_squares(
    A_reduced,
    L_reduced,
    P,
    prov_values=prov_values
)


# Individual observation testing (data snooping)
# Reuses the residuals, Qxx, and sigma0_hat already computed
# by least_squares() instead of solving the adjustment again.
# outliers at 50% confidence.
obs_test_df = test_residuals(
    A_reduced,
    P=P,
    row_labels=row_labels,
    # confidence=0.50,
    multiplier=0.6745,  # E50
    test_type='tau',
    least_squares_result=result,
    sigma0=result['sigma0'],
)


# ============================================================
# ERROR ELLIPSE
# ============================================================

Qxx = result['Qxx']

error_ellipse(
    Qxx,
    sd=0.136
)


# ============================================================
# LEVELING
# ============================================================

# Field-book data
obs = pd.DataFrame({
    "from": [
        "A", "C", "C", "D", "F",
        "A", "A", "F", "B", "B", "E"
    ],
    "to": [
        "B", "B", "D", "E", "E",
        "F", "E", "B", "E", "D", "C"
    ],
    "dh": [
        24.66, -53.36, -27.09, -33.50, 21.86,
        -16.52, 5.35, 28.86, -8.93, 26.51, 60.36
    ],
    "length": [
        5, 15, 8, 10, 7,
        6, 4, 8, 5, 8, 16
    ],
})


# Control stations and their heights
control = {
    "A": 260.00,
    "C": 325.69
}


# Provisional heights of unknown stations
provisional_heights = {
    "B": 272.33,
    "D": 298.60,
    "E": 265.10,
    "F": 243.48
}


# Run adjustment
result = leveling_adjustment(
    obs,
    control,
    provisional_heights=provisional_heights,
    multiplier=0.6745   #E50
)

ed = elevation_difference(
    result,
    "A",
    "B"
)

print(
    f"\nAdjusted dH A->B = "
    f"{ed['diff']:.4f} +/- {ed['std_error']:.4f}"
)


# ============================================================
# CORRELATION
# ============================================================

print("=" * 30)
print("CORRELATION")
print("=" * 30)

x = (6, 8.2, 10, 15, 18)
y = (13.25, 10.05, 22, 8, 10.95)

result = correlation(x, y)

# Pearson's r
result['r']

# Slope and intercept of y = ax + b
result['a'], result['b']

# Estimate y at a new x
result['predict'](12)


# ============================================================
# TRANSFORMATION
# ============================================================

print("=" * 30)
print("TRANSFORMATION")
print("=" * 30)

E_s = [
    665652.962,
    665367.934,
    658801.809,
    658773.501
]

N_s = [
    738830.245,
    738883.035,
    739890.545,
    737411.866
]

E_T = [
    718066.14,
    717131.82,
    695605.28,
    695479.68
]

N_T = [
    730024.84,
    730201.51,
    733592.91,
    725462.41
]


# Solve for conformal transformation parameters
a, b, c, d, scale, theta = solve_conformal(
    E_s,
    N_s,
    E_T,
    N_T
)


# Compute conformal residuals and RMSE
residuals_conformal(
    E_s,
    N_s,
    E_T,
    N_T,
    a,
    b,
    c,
    d
)


# Coordinates to transform
E = [658582.179, 658602.546]
N = [738036.464, 737949.267]

E_transformed, N_transformed = transform_conformal(
    E,
    N,
    a,
    b,
    c,
    d
)