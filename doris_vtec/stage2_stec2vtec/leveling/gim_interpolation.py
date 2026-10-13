import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from ...io.utils import tai_epochs_to_utc


def get_gim_vtec(
    gim: dict,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    df: pd.DataFrame,
    *,
    observation_time_system: str = "TAI",
) -> np.ndarray:
    """Earth-fixed bilinear spatial and linear temporal GIM interpolation.

    IONEX epochs are UTC; observation epochs use TAI unless explicitly set
    to UTC. Queries outside the map support or across a gap larger than the
    IONEX declared INTERVAL return NaN. Exact map epochs need no interval.
    """
    if observation_time_system not in {"TAI", "UTC"}:
        raise ValueError("observation_time_system must be 'TAI' or 'UTC'")
    epochs = pd.to_datetime(df["obs_epoch"]).to_numpy(dtype="datetime64[ns]")
    if observation_time_system == "TAI":
        epochs = np.asarray(tai_epochs_to_utc(epochs), dtype="datetime64[ns]")
    vtec_out = np.full(len(df), np.nan)
    if not gim:
        return vtec_out

    lat = df["ipp_lat"].to_numpy(dtype=float)
    lon = ((df["ipp_lon"] + 180) % 360 - 180).to_numpy(dtype=float)
    keys = sorted(gim)
    gim_epochs = pd.DatetimeIndex(keys).to_numpy(dtype="datetime64[ns]")
    if np.any(np.diff(gim_epochs) <= np.timedelta64(0, "ns")):
        raise ValueError("GIM epochs must be unique and increasing")
    interpolators = [
        RegularGridInterpolator(
            (lat_grid, lon_grid), gim[key], bounds_error=False,
            fill_value=np.nan, method="linear",
        )
        for key in keys
    ]
    for i, t_obs in enumerate(epochs):
        if np.isnat(t_obs) or not np.isfinite(lat[i]) or not np.isfinite(lon[i]):
            continue
        idx = np.searchsorted(gim_epochs, t_obs)
        point = np.array([[lat[i], lon[i]]])
        if idx < len(gim_epochs) and gim_epochs[idx] == t_obs:
            vtec_out[i] = interpolators[idx](point)[0]
        elif 0 < idx < len(gim_epochs):
            interval_seconds = getattr(gim, "interval_seconds", None)
            if interval_seconds is None:
                raise ValueError("IONEX INTERVAL is required for temporal interpolation")
            if not np.isfinite(interval_seconds) or interval_seconds <= 0:
                raise ValueError("Invalid IONEX INTERVAL for temporal interpolation")
            t0, t1 = gim_epochs[idx - 1], gim_epochs[idx]
            gap_seconds = (t1 - t0) / np.timedelta64(1, "s")
            if gap_seconds > interval_seconds:
                continue
            v0 = interpolators[idx - 1](point)[0]
            v1 = interpolators[idx](point)[0]
            weight = (t_obs - t0) / (t1 - t0)
            vtec_out[i] = v0 + weight * (v1 - v0)
    return vtec_out
