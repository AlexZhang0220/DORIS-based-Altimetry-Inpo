import pandas as pd
import numpy as np
from numba import njit

@njit
def batch_lagrange_interp_1d(x_known, y_known, x_interp):
    """
    Perform 1D Lagrange interpolation.

    Parameters
    ----------
    x_known : ndarray
        Known sample points.
    y_known : ndarray
        Function values at x_known.
    x_interp : ndarray
        Target points for interpolation.

    Returns
    -------
    ndarray
        Interpolated values at x_interp.
    """
    n = len(x_known)
    result = np.zeros_like(x_interp)
    for j in range(len(x_interp)):
        x = x_interp[j]
        total = 0.0
        for i in range(n):
            term = y_known[i]
            for k in range(n):
                if k != i:
                    term *= (x - x_known[k]) / (x_known[i] - x_known[k])
            total += term
        result[j] = total
    return result


def add_satellite_position(
    df_obs: pd.DataFrame,
    df_sp3: pd.DataFrame,
    *,
    num_neighbors: int = 8,
) -> pd.DataFrame:
    """
    Add interpolated satellite ECEF positions to observation DataFrame
    using Lagrange interpolation.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observation DataFrame containing:
        - obs_epoch
        - receiver_clock_offset

    df_sp3 : pd.DataFrame
        Satellite orbit DataFrame containing:
        - epoch
        - x, y, z   (meters)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - sat_x, sat_y, sat_z (meters)
    """
    # --------------------------------------------------
    # Prepare satellite ephemeris (block-level)
    # --------------------------------------------------
    eph_seconds = (
        pd.to_datetime(df_sp3["epoch"])
        .astype("int64")
        .to_numpy()
        / 10**9
    )

    eph_xyz = df_sp3[["x", "y", "z"]].to_numpy()

    # --------------------------------------------------
    # Observation times (apply receiver clock offset)
    # --------------------------------------------------
    obs_time = (
        pd.to_datetime(df_obs["obs_epoch"])
        + pd.to_timedelta(df_obs["receiver_clock_offset"], unit="s")
    )
    obs_seconds = obs_time.astype("int64") / 10**9

    unique_seconds = np.unique(obs_seconds)

    sat_x_list, sat_y_list, sat_z_list = [], [], []

    # --------------------------------------------------
    # Lagrange interpolation 
    # --------------------------------------------------
    for t in unique_seconds:
        idx = np.searchsorted(eph_seconds, t)

        half = num_neighbors // 2
        start = max(0, idx - half)
        end = min(len(eph_seconds), start + num_neighbors)
        start = max(0, end - num_neighbors)

        xs = eph_seconds[start:end]
        ys_x = eph_xyz[start:end, 0]
        ys_y = eph_xyz[start:end, 1]
        ys_z = eph_xyz[start:end, 2]

        if len(xs) < 2:
            sat_x_list.append(np.nan)
            sat_y_list.append(np.nan)
            sat_z_list.append(np.nan)
        else:
            sat_x_list.append(
                batch_lagrange_interp_1d(xs, ys_x, np.array([t]))[0]
            )
            sat_y_list.append(
                batch_lagrange_interp_1d(xs, ys_y, np.array([t]))[0]
            )
            sat_z_list.append(
                batch_lagrange_interp_1d(xs, ys_z, np.array([t]))[0]
            )

    # --------------------------------------------------
    # Merge back to observation DataFrame
    # --------------------------------------------------
    interp_df = pd.DataFrame(
        {
            "obs_seconds": unique_seconds,
            "sat_x": sat_x_list,
            "sat_y": sat_y_list,
            "sat_z": sat_z_list,
        }
    )

    df_obs = df_obs.copy()
    df_obs["obs_seconds"] = obs_seconds
    df_obs = df_obs.merge(interp_df, on="obs_seconds", how="left")
    df_obs = df_obs.drop(columns=["obs_seconds"])

    return df_obs