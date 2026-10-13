import pandas as pd
import numpy as np
from numba import njit
from ...io.utils import correct_receiver_epochs

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
        - prn (one satellite)

    df_sp3 : pd.DataFrame
        Satellite orbit DataFrame containing:
        - epoch
        - prn
        - x, y, z   (meters)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - sat_x, sat_y, sat_z (meters)
    """
    df_obs = correct_receiver_epochs(df_obs)
    if df_obs.empty:
        for column in ("sat_x", "sat_y", "sat_z"):
            df_obs[column] = pd.Series(index=df_obs.index, dtype=float)
        return df_obs

    if "prn" not in df_obs or "prn" not in df_sp3:
        raise ValueError("Observation and SP3 PRNs are required.")
    prns = df_obs["prn"].drop_duplicates()
    if len(prns) != 1 or pd.isna(prns.iloc[0]):
        raise ValueError("One observation PRN is required.")
    prn = prns.iloc[0]
    df_sp3 = df_sp3.loc[df_sp3["prn"].eq(prn)].copy()
    if df_sp3.empty:
        raise ValueError(f"No SP3 orbit for {prn}.")

    df_sp3["epoch"] = pd.to_datetime(df_sp3["epoch"], errors="raise")
    if df_sp3["epoch"].isna().any() or df_sp3["epoch"].dt.tz is not None:
        raise ValueError("SP3 epochs must be finite timezone-naive labels.")
    df_sp3 = df_sp3.sort_values("epoch").reset_index(drop=True)
    if df_sp3["epoch"].duplicated().any():
        raise ValueError(f"Duplicate SP3 epochs for {prn}.")
    eph_xyz = df_sp3[["x", "y", "z"]].to_numpy(dtype=float)
    if not np.isfinite(eph_xyz).all():
        raise ValueError(f"Non-finite SP3 coordinates for {prn}.")

    # --------------------------------------------------
    # Prepare satellite ephemeris (block-level)
    # --------------------------------------------------
    eph_seconds = (
        pd.to_datetime(df_sp3["epoch"])
        .astype("int64")
        .to_numpy()
        / 10**9
    )

    # --------------------------------------------------
    # Observation times (apply receiver clock offset)
    # --------------------------------------------------
    obs_time = df_obs["obs_epoch"]
    obs_seconds = obs_time.astype("int64") / 10**9

    unique_seconds = np.unique(obs_seconds)

    sat_x_list, sat_y_list, sat_z_list = [], [], []

    # --------------------------------------------------
    # Lagrange interpolation 
    # --------------------------------------------------
    for t in unique_seconds:
        if (
            len(eph_seconds) < 2
            or t < eph_seconds[0]
            or t > eph_seconds[-1]
        ):
            sat_x_list.append(np.nan)
            sat_y_list.append(np.nan)
            sat_z_list.append(np.nan)
            continue
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
    attributes = dict(df_obs.attrs)
    df_obs["obs_seconds"] = obs_seconds
    df_obs = df_obs.merge(interp_df, on="obs_seconds", how="left")
    df_obs = df_obs.drop(columns=["obs_seconds"])
    df_obs.attrs.update(attributes)

    return df_obs
