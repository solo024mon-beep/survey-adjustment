# Survey Adjustment Toolkit — Project Documentation & Formula Reference

A Python toolkit for least-squares survey computations: network adjustment
(leveling), error ellipses, residual/blunder testing, coordinate
transformations, and descriptive statistics. Every formula below is
described with its parameters, and matches the code exactly (module and
line references are given where useful).

---

## Part A — Project Documentation

### A.1 Module overview

| Module | Purpose |
|---|---|
| `leveling_adjustment.py` | Least-squares adjustment of a differential leveling (height) network |
| `errorellipse.py` | Standard error ellipse for one or more adjusted points, from a cofactor matrix |
| `residual_test.py` | Per-observation blunder detection (Baarda's w-test / Pope's tau-test) |
| `survey_stats.py` | Descriptive statistics, probable-error outlier screening, and Pearson correlation |
| `transformation.py` | 2D conformal & affine coordinate transformations, and geodetic ⇄ ECEF conversion |

All modules use `numpy` for linear algebra, `pandas` for labeled
tabular I/O, and (where a statistical critical value is needed)
`scipy.stats`.

---

### A.2 `leveling_adjustment.py`

**Function:** `leveling_adjustment(obs, control, from_col='from', to_col='to', dh_col='dh', length_col='length', std_col=None, c=1.0, sigma0_apriori=1.0, alpha=0.05, verbose=True)`

Runs a full weighted least-squares adjustment of a differential leveling
network.

- **Input:** a `pandas.DataFrame` of leveling runs — columns `from`,
  `to`, `dh` (observed elevation change, *to* minus *from*), and either
  `length` (weight ∝ 1/length) or `std` (weight = 1/std²). A `control`
  dict of `{point_label: known_elevation}` marks fixed benchmarks; every
  other point becomes an unknown to solve for.
- **What it does:**
  1. Builds the design matrix `A`, observation vector `L`, and weight
     matrix `P` (see §1 and §2 of the formula sheet for the exact
     construction rules).
  2. Solves the normal equations for adjusted heights.
  3. Computes residuals, redundancy, a-posteriori σ₀, the cofactor
     matrices `Qxx`/`Qll`/`Qvv`, and standard errors of both the
     adjusted heights and the residuals.
  4. Runs Baarda's w-test on every observation (§4.2) and a global
     chi-square model test (§4.4).
  5. Separates out any control-to-control runs (both endpoints fixed)
     as a misclosure check, since they carry no information about any
     unknown.
- **Returns:** a dict containing `A`, `L`, `P`, `x`, `adjusted_heights`,
  `v`, `sigma0`, `Qxx`, `Qll`, `Qvv`, `std_heights`, `std_residuals`,
  `standardized_resid`, `flagged`, `redundancy`, `chi2_stat`, and more
  (see the function's docstring for the full key list).

**Function:** `elevation_difference(result, point1, point2)`

Given the dict returned above, estimates the adjusted elevation
difference between any two stations (control or adjusted-unknown) with
its propagated standard error (§1.9 / law of variance propagation).

---

### A.3 `errorellipse.py`

**Function:** `error_ellipse(Qxx, S0=1.0, point_labels=None, verbose=True)`

Computes the standard error ellipse for one or more points from their
cofactor matrix.

- **Input:** `Qxx` — either a single point's 2×2 cofactor sub-block, or
  the full `2n × 2n` cofactor matrix for `n` points (unknowns ordered
  `[E1, N1, E2, N2, ...]`); only the block-diagonal 2×2 sub-matrices are
  used. `S0` is a single scale multiplier — fold in σ₀ and/or a
  confidence-level factor as needed (§5.3).
- **Returns:** a `pandas.DataFrame` with one row per point:
  `w`, `a`, `b`, `Sx`, `Sy`, `theta_deg`, `Bearing` (see §5 for every
  formula).
- Note: `qxx`/`qyy` are allowed to be negative in the input (not a
  physically valid cofactor block, but the `a`/`theta`/`Bearing`
  formulas don't break); `b`/`Sx`/`Sy` come back `NaN` rather than
  crashing when their underlying square root would be negative.

---

### A.4 `residual_test.py`

**Function:** `test_residuals(A, L=None, P=None, row_labels=None, alpha=0.05, confidence=None, sigma0=1.0, test_type='w', verbose=True, least_squares_result=None)`

Tests every observation's residual against a critical value following a
weighted least-squares adjustment (§4.2–4.3).

- Can either **compute the adjustment itself** (pass `A`, `L`, `P`), or
  **reuse an already-computed one** (pass `least_squares_result`, the
  dict from an external `least_squares()`/`leveling_adjustment()` call,
  to avoid re-solving and guarantee agreement).
- `test_type='w'` — Baarda's test, assumes σ₀ known a priori (normal
  distribution critical value).
- `test_type='tau'` — Pope's test, uses the a-posteriori σ̂₀ (Student's
  t critical value); the more defensible default when σ₀ is estimated
  from the same adjustment, which is the normal case in practice.
- **Returns:** a `pandas.DataFrame` with columns `Line`, `Residual`,
  `Sd-Residual`, `Value`, `Rc`, `Accepted/Rejected` — `Value` and `Rc`
  are both dimensionless (standardized) so they're directly comparable
  left-to-right in the table.

---

### A.5 `survey_stats.py`

**Function:** `statistics(data, verbose=True)`

Descriptive statistics and probable-error outlier screening for a set
of repeated observations (§7, §8): mean, median, mode, range, variance,
standard deviation, standard error of the mean, the E50/E90/E99
probable-error intervals, and an outlier flag.

**Function:** `correlation(x, y, verbose=True)`

Pearson correlation coefficient and simple linear regression
(line-of-best-fit) for paired observations (§9).

- **Returns:** a dict including `covariance`, `Sx`, `Sy`, `r`,
  `r_squared`, `strength` (a plain-language description of `|r|`), the
  best-fit line's `a`/`b`, and a `predict(x_new)` callable.

---

### A.6 `transformation.py`

**Function:** `solve_conformal(E_s, N_s, E_T, N_T, verbose=True)`

Solves the 2D conformal (Helmert) transformation by least squares
(§6.1). Prints the design matrix `A` and observation vector `L`, then
the solved `a, b, c, d, scale, theta`.

**Function:** `transform_conformal(E, N, a, b, c, d)`

Applies a solved conformal transformation to new coordinates.

**Function:** `residuals_conformal(E_s, N_s, E_T, N_T, a, b, c, d)`

Computes per-point residuals and RMSE for a solved conformal
transformation (§6.3).

**Function:** `affine_transformation(E_s, N_s, E_T, N_T, E_new=None, N_new=None, verbose=True)`

Solves the 2D affine transformation by least squares (§6.2; needs ≥3
non-collinear control points), and optionally transforms new points in
the same call.

**Function:** `geodetic_to_ecef(lat_str, lon_str, height)` /
`ecef_to_geodetic(X, Y, Z)`

Converts between geodetic (lat/lon/height, WGS84) and ECEF (X, Y, Z)
coordinates (§6.4). `lat_str`/`lon_str` are `"deg min sec"` strings
(negative degrees for southern latitude / western longitude are
handled correctly).

---
---

## Part B — Formula Sheet

Every formula lists its symbols and what each one means directly
underneath it.

---

### 1. Least-Squares Adjustment (core)

**1.1 Normal equations**

$$N = A^T P A \qquad U = A^T P L$$

| Symbol | Meaning |
|---|---|
| $A$ | Design (coefficient) matrix, shape $(n_{obs}, n_{unknowns})$ |
| $P$ | Weight matrix, shape $(n_{obs}, n_{obs})$ |
| $L$ | Observation vector |
| $N$ | Normal matrix |
| $U$ | Normal equation vector |

**1.2 Solution**

$$\hat{x} = N^{-1}U = (A^TPA)^{-1}A^TPL$$

| Symbol | Meaning |
|---|---|
| $\hat{x}$ | Adjusted unknowns (or corrections to provisional unknowns, depending on model — see §1.6) |

**1.3 Residuals**

$$V = A\hat{x} - L$$

| Symbol | Meaning |
|---|---|
| $V$ | Vector of residuals, one per observation |

**1.4 Redundancy (degrees of freedom)**

$$r = n - u$$

| Symbol | Meaning |
|---|---|
| $n$ | Number of observations |
| $u$ | Number of unknowns |
| $r$ | Redundancy — number of "extra" observations beyond the minimum needed to solve the network; must be $> 0$ for residuals/statistics to be meaningful |

**1.5 A-posteriori variance factor**

$$\hat{\sigma}_0^{2} = \frac{V^TPV}{r} \qquad \hat{\sigma}_0 = \sqrt{\hat{\sigma}_0^{2}}$$

| Symbol | Meaning |
|---|---|
| $\hat{\sigma}_0$ | A-posteriori standard deviation of unit weight — how well the actual data fits, given the stated weights |

**1.6 Final adjusted values (when solving for corrections to provisional values, rather than the unknowns directly)**

$$X_{final} = X_{provisional} + \hat{x}$$

*(The leveling and transformation modules in this project solve directly
for the unknowns — heights or transformation parameters — so this step
isn't needed there; it applies if you're linearizing around an
approximate value, e.g. in a nonlinear traverse adjustment.)*

---

### 2. Weight Matrix & Cofactor Matrices

**2.1 Weight from standard deviation**

$$P_{ii} = \left(\frac{\sigma_0}{\sigma_i}\right)^{2}$$

| Symbol | Meaning |
|---|---|
| $\sigma_0$ | A-priori reference standard deviation of unit weight (often 1) |
| $\sigma_i$ | Standard deviation of observation $i$ |
| $P_{ii}$ | Weight of observation $i$ |

If $\sigma_0 = 1$: $P_{ii} = 1/\sigma_i^2$. As implemented in
`leveling_adjustment.py`, when a run's `std` isn't given directly,
weight is instead taken proportional to run length:

$$P_{ii} = \frac{c}{\text{length}_i}$$

| Symbol | Meaning |
|---|---|
| $c$ | Arbitrary weighting constant (only *relative* weights matter; any convenient value works, e.g. the shortest run's length) |
| $\text{length}_i$ | Length of leveling run $i$ |

**2.2 Cofactor matrix of the observations**

$$Q_{ll} = P^{-1}$$

**2.3 Cofactor matrix of the unknowns**

$$Q_{xx} = (A^TPA)^{-1} = N^{-1}$$

**2.4 Covariance matrix of the unknowns**

$$C_{xx} = \hat{\sigma}_0^{2}\,Q_{xx}$$

**2.5 Standard deviations of the unknowns**

$$\sigma_{x_i} = \hat{\sigma}_0\sqrt{(Q_{xx})_{ii}} = \sqrt{(C_{xx})_{ii}}$$

| Symbol | Meaning |
|---|---|
| $(Q_{xx})_{ii}$ | Diagonal entry of $Q_{xx}$ for unknown $i$ |
| $\sigma_{x_i}$ | Standard error of adjusted unknown $i$ |

**2.6 Residual cofactor matrix**

$$Q_{vv} = Q_{ll} - AQ_{xx}A^T$$

| Symbol | Meaning |
|---|---|
| $Q_{vv}$ | Cofactor matrix of the residuals; its diagonal gives each residual's own variance |

---

### 3. Leveling Network Observation Equations

For a run from station $F$ to station $T$ with observed elevation
difference $dh$ (i.e. $T$ minus $F$, as normally recorded in a leveling
field book), the model is:

$$dh_{obs} + v = H_T - H_F$$

| Symbol | Meaning |
|---|---|
| $dh_{obs}$ | Observed elevation change from $F$ to $T$ |
| $v$ | Residual for this observation |
| $H_F, H_T$ | (Adjusted or fixed) elevations of $F$ and $T$ |

This is rearranged into $A\hat{x} = L$ form differently depending on
which endpoints are unknown vs. fixed control:

| Case | Row of $A$ | $L$ |
|---|---|---|
| Both $F$, $T$ unknown | $-1$ at $F$, $+1$ at $T$ | $dh_{obs}$ |
| $F$ unknown, $T$ is control | $+1$ at $F$ | $H_T - dh_{obs}$ |
| $T$ unknown, $F$ is control | $+1$ at $T$ | $dh_{obs} + H_F$ |
| Both $F$, $T$ control | *(excluded from the adjustment — reported as a misclosure check instead)* | — |

---

### 4. Residual Testing / Data Snooping

**4.1 Baarda's w-test** (a-priori $\sigma_0$ known)

$$w_i = \frac{v_i}{\sigma_0\sqrt{(Q_{vv})_{ii}}} \qquad R_c = z_{1-\alpha/2}$$

| Symbol | Meaning |
|---|---|
| $v_i$ | Residual of observation $i$ |
| $\sigma_0$ | A-priori standard deviation of unit weight (default 1.0) |
| $(Q_{vv})_{ii}$ | Diagonal entry of $Q_{vv}$ for observation $i$ |
| $\alpha$ | Two-tailed significance level (e.g. 0.05) |
| $z_{1-\alpha/2}$ | Standard normal critical value (≈1.96 for $\alpha=0.05$) |

Decision: $|w_i| > R_c \Rightarrow$ **Rejected** (likely blunder), else
**Accepted**.

**4.2 Pope's tau-test** (a-posteriori $\hat{\sigma}_0$ estimated)

$$\tau_i = \frac{v_i}{\hat{\sigma}_0\sqrt{(Q_{vv})_{ii}}} \qquad R_c = t_{1-\alpha/2,\ r-1}$$

| Symbol | Meaning |
|---|---|
| $\hat{\sigma}_0$ | A-posteriori standard deviation of unit weight (from the adjustment itself) |
| $t_{1-\alpha/2,\ r-1}$ | Student's t critical value with $r-1$ degrees of freedom |

Use this instead of the w-test whenever $\sigma_0$ is *not* known in
advance and must be estimated from the same adjustment — the normal
case for real survey data.

**4.3 Global (overall) model test — chi-square**

$$\chi^{2} = r \cdot \frac{\hat{\sigma}_0^2}{\sigma_0^2} \sim \chi^2_r$$

| Symbol | Meaning |
|---|---|
| $r$ | Redundancy |
| $\hat{\sigma}_0^2$ | A-posteriori variance |
| $\sigma_0^2$ | A-priori variance (assumed going in) |

Compare against the acceptance range
$[\chi^2_{\alpha/2,\,r},\ \chi^2_{1-\alpha/2,\,r}]$. Tests the
adjustment as a whole rather than any single observation — see the
"chi-square vs. Baarda" discussion in the project notes for how the two
tests complement each other.

---

### 5. Error Ellipse

Given a point's own 2×2 cofactor sub-block:

$$Q_{point} = \begin{bmatrix} q_{xx} & q_{xy} \\ q_{xy} & q_{yy} \end{bmatrix}$$

| Symbol | Meaning |
|---|---|
| $q_{xx}, q_{yy}$ | Diagonal cofactors — variance-like terms for the point's E and N coordinates |
| $q_{xy}$ | Off-diagonal cofactor — covariance-like term between E and N |

**5.1 Intermediate radical term**

$$w = \sqrt{(q_{xx}-q_{yy})^2 + 4q_{xy}^2}$$

**5.2 Semi-major / semi-minor axes**

$$a = S_0\sqrt{\tfrac{1}{2}(q_{xx}+q_{yy}+w)} \qquad b = S_0\sqrt{\tfrac{1}{2}(q_{xx}+q_{yy}-w)}$$

| Symbol | Meaning |
|---|---|
| $S_0$ | Scale multiplier — fold in $\hat{\sigma}_0$ and/or a confidence-level factor $k$ (§5.5) |
| $a$ | Semi-major axis length |
| $b$ | Semi-minor axis length |

**5.3 Standard error along the E/N axes directly (unrotated)**

$$S_x = S_0\sqrt{q_{xx}} \qquad S_y = S_0\sqrt{q_{yy}}$$

**5.4 Orientation**

$$\tan(2\theta) = \frac{2q_{xy}}{q_{xx}-q_{yy}} \qquad \theta = \tfrac{1}{2}\operatorname{atan2}(2q_{xy},\ q_{xx}-q_{yy})$$

| Symbol | Meaning |
|---|---|
| $\theta$ | Orientation of the major axis, measured mathematically from the E axis |

Converted to a surveying bearing (0–360°, clockwise from North) by a
quadrant correction based on the signs of $2q_{xy}$ and $q_{xx}-q_{yy}$
(see `error_ellipse()` for the exact branches).

**5.5 Scaling to a confidence level**

$$k = \sqrt{\chi^2_{p,\,2}} \qquad a_{scaled} = k\cdot a \qquad b_{scaled} = k \cdot b$$

| Symbol | Meaning |
|---|---|
| $p$ | Desired confidence level (e.g. 0.95) |
| $\chi^2_{p,\,2}$ | Chi-square quantile with 2 degrees of freedom |
| $k$ | Scale factor — common values: $k=1.0$ (~39.4%, the "standard" ellipse), $k\approx2.146$ (95%), $k\approx3.035$ (99%) |

---

### 6. Coordinate Transformations

**6.1 Conformal (Helmert) — 4 parameters**

$$E' = aE - bN + c \qquad N' = bE + aN + d$$

| Symbol | Meaning |
|---|---|
| $E, N$ | Source coordinates |
| $E', N'$ | Target (transformed) coordinates |
| $a, b$ | Scale/rotation parameters |
| $c, d$ | Translation parameters (in E and N respectively) |

Recovering scale and rotation:

$$s = \sqrt{a^2+b^2} \qquad \theta = \operatorname{atan2}(b,a)$$

| Symbol | Meaning |
|---|---|
| $s$ | Uniform scale factor |
| $\theta$ | Rotation angle |

Design matrix (per control point, 2 rows):

$$\begin{bmatrix}E'\\N'\end{bmatrix} = \begin{bmatrix}E & -N & 1 & 0\\N & E & 0 & 1\end{bmatrix}\begin{bmatrix}a\\b\\c\\d\end{bmatrix}$$

Minimum 2 common points (4 equations for 4 unknowns); more points give
redundancy for a least-squares solution via §1.2.

**6.2 Affine — 6 parameters**

$$E' = aE + bN + c \qquad N' = dE + eN + f$$

| Symbol | Meaning |
|---|---|
| $a, b, d, e$ | Independent scale/shear parameters (angles are *not* preserved) |
| $c, f$ | Translation parameters |

Design matrix (per control point, 2 rows):

$$\begin{bmatrix}E'\\N'\end{bmatrix} = \begin{bmatrix}E & N & 1 & 0 & 0 & 0\\0 & 0 & 0 & E & N & 1\end{bmatrix}\begin{bmatrix}a\\b\\c\\d\\e\\f\end{bmatrix}$$

Minimum 3 non-collinear common points (6 equations for 6 unknowns).

**6.3 Transformation residuals and fit quality**

$$v_E = E_T - E_{calc} \qquad v_N = N_T - N_{calc}$$

$$RMSE = \sqrt{\frac{\sum(v_E^2+v_N^2)}{n}}$$

| Symbol | Meaning |
|---|---|
| $E_T, N_T$ | Target (known/observed) coordinates of a control point |
| $E_{calc}, N_{calc}$ | Coordinates computed by applying the solved transformation to the source point |
| $n$ | Number of common (control) points used |

**6.4 Geodetic ⇄ ECEF (WGS84)**

Ellipsoid constants:

$$a = 6378137.0\text{ m} \qquad f = 1/298.257223563 \qquad e = \sqrt{2f - f^2}$$

| Symbol | Meaning |
|---|---|
| $a$ | Semi-major axis of the WGS84 ellipsoid |
| $f$ | Flattening |
| $e$ | First eccentricity |

Prime vertical radius of curvature:

$$\nu = \frac{a}{\sqrt{1-e^2\sin^2\phi}}$$

| Symbol | Meaning |
|---|---|
| $\phi$ | Geodetic latitude |
| $\nu$ | Radius of curvature in the prime vertical |

Geodetic → ECEF:

$$X = (\nu+h)\cos\phi\cos\lambda \qquad Y = (\nu+h)\cos\phi\sin\lambda \qquad Z = \big((1-e^2)\nu+h\big)\sin\phi$$

| Symbol | Meaning |
|---|---|
| $\lambda$ | Geodetic longitude |
| $h$ | Height above the ellipsoid |
| $X, Y, Z$ | ECEF coordinates |

ECEF → Geodetic is solved iteratively (5 iterations, converges quickly
for terrestrial heights):

$$\lambda = \operatorname{atan2}(Y,X) \qquad p = \sqrt{X^2+Y^2}$$

$$\phi_0 = \operatorname{atan2}\!\big(Z,\ p(1-e^2)\big)$$

then repeat: $\nu = a/\sqrt{1-e^2\sin^2\phi}$,
$h = p/\cos\phi - \nu$,
$\phi \leftarrow \operatorname{atan2}\!\big(Z+e^2\nu\sin\phi,\ p\big)$.

**Note on DMS input:** latitude/longitude strings like `"43 15 46.289"`
are converted to decimal degrees as
$\phi = \text{sign}(\deg)\cdot(|\deg| + \min/60 + \sec/3600)$ — the sign
must be applied to the *whole* magnitude, not just the degree part, or
southern latitudes / western longitudes convert incorrectly.

---

### 7. Descriptive Statistics

For a dataset of $n$ observations $x_1,\dots,x_n$:

**7.1 Mean** $\quad \bar{x} = \dfrac{\sum x_i}{n}$

**7.2 Median** — middle value when sorted (average of the two middle
values if $n$ is even).

**7.3 Mode** — the most frequently occurring value(s).

**7.4 Range** $\quad R = x_{max}-x_{min}$

**7.5 Variance (sample)** $\quad s^2 = \dfrac{\sum(x_i-\bar{x})^2}{n-1}$

**7.6 Standard deviation (sample)** $\quad s = \sqrt{s^2}$

**7.7 Standard error of the mean** $\quad SE_{\bar{x}} = \dfrac{s}{\sqrt{n}}$

| Symbol | Meaning |
|---|---|
| $n$ | Number of observations |
| $s^2, s$ | Sample variance, sample standard deviation |
| $SE_{\bar{x}}$ | Standard error of the mean |

---

### 8. Probable Error Intervals & Rejection Criteria

**8.1 Probable error at confidence level $p$**

$$E_p = z_p \cdot s$$

| Confidence | Multiplier $z_p$ |
|---|---|
| 50% ($E_{50}$) | 0.6745 |
| 90% ($E_{90}$) | 1.6449 |
| 95% | 1.9600 |
| 99% ($E_{99}$) | 2.5758 |

**8.2 Confidence interval** $\quad \bar{x}-E_p < x < \bar{x}+E_p$

**8.3 Consistency check**

$$\text{Percentage} = \frac{\text{count within interval}}{n}\times100\%$$

**8.4 Simple outlier / rejection test**

$$|x_i-\bar{x}| > R_c$$

| Symbol | Meaning |
|---|---|
| $R_c$ | Typically the $E_{99}$ half-width ($R_c=2.576\,s$); match the multiplier to whatever confidence level the question specifies |

**8.5 Relationship between confidence and significance level**

$$\alpha = 1-\text{confidence}$$

$$R_c = z_{1-\alpha/2}\ \text{(normal — w-test / simple outlier test)} \qquad R_c = t_{1-\alpha/2,\ r-1}\ \text{(Student's t — tau-test)}$$

---

### 9. Correlation & Line of Best Fit (Pearson)

For paired observations $(x_i, y_i)$, $i=1,\dots,n$:

**9.1 Deviations from the mean**

$$R_{x,i} = x_i-\bar{x} \qquad R_{y,i} = y_i-\bar{y}$$

**9.2 Covariance**

$$\text{cov}(x,y) = \frac{\sum R_{x,i}R_{y,i}}{n-1}$$

**9.3 Standard deviations**

$$S_x = \sqrt{\frac{\sum R_{x,i}^2}{n-1}} \qquad S_y = \sqrt{\frac{\sum R_{y,i}^2}{n-1}}$$

**9.4 Pearson's correlation coefficient**

$$r = \frac{\text{cov}(x,y)}{S_x\,S_y} \qquad r^2 = \text{coefficient of determination}$$

| Symbol | Meaning |
|---|---|
| $r$ | Pearson correlation coefficient, $-1\le r\le 1$; sign gives direction, magnitude gives strength |
| $r^2$ | Proportion of variance in $y$ "explained" by a linear relationship with $x$ |

**9.5 Line of best fit**

$$y = ax+b \qquad a = \frac{r\,S_y}{S_x} \qquad b = \bar{y}-a\bar{x}$$

| Symbol | Meaning |
|---|---|
| $a$ | Slope of the best-fit line |
| $b$ | Intercept of the best-fit line |

---

## 10. Quick-Reference Summary Table

| Quantity | Formula |
|---|---|
| Normal matrix | $N=A^TPA$ |
| Normal equation vector | $U=A^TPL$ |
| Adjusted unknowns | $\hat{x}=N^{-1}U$ |
| Residuals | $V=A\hat{x}-L$ |
| Redundancy | $r=n-u$ |
| A-posteriori variance | $\hat{\sigma}_0^2=V^TPV/r$ |
| Cofactor matrix (unknowns) | $Q_{xx}=N^{-1}$ |
| Covariance matrix (unknowns) | $C_{xx}=\hat{\sigma}_0^2Q_{xx}$ |
| Std. dev. of unknowns | $\sigma_{x_i}=\sqrt{(C_{xx})_{ii}}$ |
| Cofactor matrix (observations) | $Q_{ll}=P^{-1}$ |
| Residual cofactor matrix | $Q_{vv}=Q_{ll}-AQ_{xx}A^T$ |
| w-test statistic | $w_i=v_i/(\sigma_0\sqrt{(Q_{vv})_{ii}})$ |
| tau-test statistic | $\tau_i=v_i/(\hat{\sigma}_0\sqrt{(Q_{vv})_{ii}})$ |
| Global chi-square test | $\chi^2=r\hat{\sigma}_0^2/\sigma_0^2$ |
| Error ellipse radical | $w=\sqrt{(q_{xx}-q_{yy})^2+4q_{xy}^2}$ |
| Error ellipse axes | $a,b=S_0\sqrt{\tfrac12(q_{xx}+q_{yy}\pm w)}$ |
| Error ellipse orientation | $\tan(2\theta)=2q_{xy}/(q_{xx}-q_{yy})$ |
| Conformal transform | $E'=aE-bN+c,\ \ N'=bE+aN+d$ |
| Affine transform | $E'=aE+bN+c,\ \ N'=dE+eN+f$ |
| Transformation RMSE | $\sqrt{\sum(v_E^2+v_N^2)/n}$ |
| Mean | $\bar{x}=\sum x_i/n$ |
| Sample variance | $s^2=\sum(x_i-\bar{x})^2/(n-1)$ |
| Standard error of mean | $SE=s/\sqrt{n}$ |
| Probable error | $E_p=z_p\cdot s$ |
| Pearson's r | $r=\text{cov}(x,y)/(S_xS_y)$ |
| Best-fit line | $a=rS_y/S_x,\ \ b=\bar{y}-a\bar{x}$ |


---

## 11. Additional Formulas Used by `fincomp.py` Dependencies

This section contains formulas implemented by the modules called directly by
`fincomp.py` that were not explicitly written in the original formula sheet.
The main program calls these routines through imports such as
`least_squares()`, `elevation_difference()`, `error_ellipse()`,
`solve_conformal()`, and `residuals_conformal()`. These formulas therefore
form part of the mathematical workflow used by `fincomp.py`.

### 11.1 General observation model

The leveling adjustment is implemented in the indirect/parametric form:

$$
L + V = AX
$$

or equivalently,

$$
V = AX-L
$$

where $X$ contains the unknown parameters or corrections.

### 11.2 Provisional-height / correction model

When provisional heights are supplied, the unknown vector contains
corrections rather than final heights.

For observation $i$:

$$
L_i =
dh_{obs,i}
-
\left(H_{T,prov,i}-H_{F,prov,i}\right)
$$

where the control station height is used as the exact height when an endpoint
is a control point.

The final adjusted height is:

$$
H_{final,i}=H_{prov,i}+\hat{x}_i
$$

This is the formulation used by the leveling adjustment when
`provisional_heights` is supplied.

### 11.3 Variance of an adjusted unknown

The variance of adjusted unknown $x_i$ is:

$$
\sigma_{x_i}^{2}
=
\hat{\sigma}_0^2(Q_{xx})_{ii}
$$

Equivalently,

$$
\sigma_{x_i}^{2}
=
(C_{xx})_{ii}
$$

and therefore:

$$
\sigma_{x_i}
=
\sqrt{\hat{\sigma}_0^2(Q_{xx})_{ii}}
$$

For the zero-redundancy case, the implementation uses the a-priori
variance factor instead:

$$
\sigma_{x_i}^{2}
=
\sigma_{0,apriori}^{2}(Q_{xx})_{ii}
$$

### 11.4 Standard deviation of an individual residual

The standard deviation assigned to residual $v_i$ is:

$$
\sigma_{v_i}
=
\hat{\sigma}_0\sqrt{(Q_{vv})_{ii}}
$$

The standardized residual used for data snooping is then:

$$
w_i
=
\frac{v_i}{\sigma_{v_i}}
=
\frac{v_i}
{\hat{\sigma}_0\sqrt{(Q_{vv})_{ii}}}
$$

For the a-priori Baarda formulation, this may instead be written as:

$$
w_i
=
\frac{v_i}
{\sigma_0\sqrt{(Q_{vv})_{ii}}}
$$

### 11.5 Adjusted elevation difference between two stations

For two stations $1$ and $2$, the adjusted elevation difference is:

$$
\Delta H_{12}=H_2-H_1
$$

If both stations are adjusted unknowns, its propagated variance is:

$$
\sigma_{\Delta H}^{2}
=
\hat{\sigma}_0^2
\left[
q_{11}+q_{22}-2q_{12}
\right]
$$

where

$$
q_{11}=(Q_{xx})_{11},
\qquad
q_{22}=(Q_{xx})_{22},
\qquad
q_{12}=(Q_{xx})_{12}
$$

The propagated standard error is:

$$
\sigma_{\Delta H}
=
\sqrt{
\hat{\sigma}_0^2
\left[
q_{11}+q_{22}-2q_{12}
\right]
}
$$

For a control point, its variance and covariance contributions are treated
as zero because the control height is considered exact.

### 11.6 Error-ellipse bearing conversion

The mathematical orientation of the major axis is:

$$
\theta
=
\frac{1}{2}
\operatorname{atan2}
\left(
2q_{xy},
q_{xx}-q_{yy}
\right)
$$

The implementation converts this mathematical angle to a surveying bearing
(clockwise from North) using:

$$
Bearing=(90^\circ-\theta)\bmod 360^\circ
$$

This is the direct bearing conversion used by the error-ellipse routine.

### 11.7 Conformal transformation solved by ordinary least squares

The conformal transformation is written as:

$$
AX=L
$$

with

$$
X=
\begin{bmatrix}
a\\b\\c\\d
\end{bmatrix}
$$

For the unweighted transformation implementation, the least-squares
solution is:

$$
\hat{X}
=
(A^TA)^{-1}A^TL
$$

when $A^TA$ is nonsingular.

The code obtains the equivalent least-squares solution using
`numpy.linalg.lstsq()`.

### 11.8 Affine transformation solved by ordinary least squares

For the affine transformation:

$$
AX=L
$$

where

$$
X=
\begin{bmatrix}
a\\b\\c\\d\\e\\f
\end{bmatrix}
$$

The ordinary least-squares solution is:

$$
\hat{X}
=
(A^TA)^{-1}A^TL
$$

when $A^TA$ is nonsingular.

Again, the implementation obtains the least-squares solution using
`numpy.linalg.lstsq()`.

### 11.9 Separate transformation RMSE components

The conformal transformation module calculates three RMSE values.

Easting RMSE:

$$
RMSE_E
=
\sqrt{
\frac{\sum_{i=1}^{n}v_{E,i}^{2}}{n}
}
$$

Northing RMSE:

$$
RMSE_N
=
\sqrt{
\frac{\sum_{i=1}^{n}v_{N,i}^{2}}{n}
}
$$

Overall coordinate RMSE:

$$
RMSE_{total}
=
\sqrt{
\frac{\sum_{i=1}^{n}
\left(v_{E,i}^{2}+v_{N,i}^{2}\right)}
{n}
}
$$

where

$$
v_{E,i}=E_{T,i}-E_{calc,i}
$$

and

$$
v_{N,i}=N_{T,i}-N_{calc,i}
$$

### 11.10 Conformal transformation rotation in degrees

The transformation module calculates the rotation as:

$$
\theta
=
\operatorname{atan2}(b,a)
$$

and converts it to degrees:

$$
\theta_{deg}
=
\frac{180}{\pi}
\operatorname{atan2}(b,a)
$$

The uniform scale factor is:

$$
s=\sqrt{a^2+b^2}
$$

### 11.11 Affine transformation equations for transformed points

After solving the six affine parameters, a new point is transformed using:

$$
E_{new}
=
aE+bN+c
$$

$$
N_{new}
=
dE+eN+f
$$

### 11.12 Conformal transformation equations for transformed points

After solving the four conformal parameters:

$$
E_{new}
=
aE-bN+c
$$

$$
N_{new}
=
bE+aN+d
$$

### 11.13 Confidence-to-significance conversion used by residual testing

When confidence rather than significance is supplied:

$$
\alpha=1-\text{confidence}
$$

For the `fincomp.py` call:

$$
\text{confidence}=0.50
$$

so:

$$
\alpha=1-0.50=0.50
$$

The corresponding two-sided critical probability is:

$$
1-\frac{\alpha}{2}
$$

and for Pope's tau-test the critical value is obtained from:

$$
R_c
=
t_{1-\alpha/2,\;r-1}
$$

### 11.14 Global chi-square acceptance limits

The global test statistic is:

$$
\chi^2_{calc}
=
r
\frac{\hat{\sigma}_0^2}
{\sigma_{0,apriori}^2}
$$

The lower and upper critical limits are:

$$
\chi^2_{low}
=
\chi^2_{\alpha/2,\;r}
$$

and

$$
\chi^2_{high}
=
\chi^2_{1-\alpha/2,\;r}
$$

The model passes when:

$$
\chi^2_{low}
\leq
\chi^2_{calc}
\leq
\chi^2_{high}
$$

### 11.15 Summary of formulas added

The additional formulas above cover the dependency calculations that were
not explicitly represented in the original formula sheet:

1. $L+V=AX$
2. Provisional-height misclosure equation
3. Final height = provisional height + correction
4. Variance of an adjusted unknown
5. Standard deviation of an adjusted unknown
6. Standard deviation of a residual
7. Standardized residual from residual standard deviation
8. Adjusted elevation difference
9. Propagated variance of an elevation difference
10. Propagated standard error of an elevation difference
11. Error-ellipse mathematical-angle to surveying-bearing conversion
12. Ordinary least-squares solution for conformal transformation
13. Ordinary least-squares solution for affine transformation
14. Easting RMSE
15. Northing RMSE
16. Overall transformation RMSE
17. Rotation conversion from radians to degrees
18. Confidence-to-significance relationship used by the tau-test
19. Explicit chi-square lower/upper critical limits
20. Chi-square acceptance condition
