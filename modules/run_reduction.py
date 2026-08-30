"""
run_reduction.py

Run this to:
  1. Be prompted for which observation type(s) to use (distance-only,
     bearing-only, or both) and which station(s) are control points.
  2. Get back ONE design matrix (A), ONE observation vector (L), and
     ONE weight matrix (P) -- already filtered to that observation-type
     choice and with control-point columns already excluded -- ready to
     feed straight into your least-squares adjustment.

Edit the data block below to match whichever network you're working
with (this one is copied from fincomp.py).
"""

from control_point_reducer import interactive_reduce
from fincomp import Lines,obs_bearing,cal_bearing,obs_dist,cal_dist
# ── Network data (same as fincomp.py) ───────────────────────────────────────


if __name__ == "__main__":
    A, L, P, unknown_stations, col_labels, row_labels = interactive_reduce(
        Lines, obs_bearing, cal_bearing, obs_dist, cal_dist
    )

    # A, L, P are numpy arrays, all built ONCE and already filtered to the
    # observation type(s) you chose and reduced to the free stations --
    # use them directly for the normal equations of your least-squares
    # adjustment, e.g.:
    #   N = A.T @ P @ A
    #   U = A.T @ P @ L
    #   delta = np.linalg.solve(N, U)
