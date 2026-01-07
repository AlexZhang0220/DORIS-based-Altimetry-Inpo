import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def get_gim_vtec(
    gim: dict,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Interpolate GIM VTEC at given IPP locations and epochs.

    Parameters
    ----------
    gim : dict[datetime -> 2D np.ndarray]
        GIM TEC maps (lat, lon) in TECU.
    lat_grid, lon_grid : np.ndarray
        Latitude / longitude grids.
    df : pd.DataFrame
        Must contain 'obs_epoch', 'ipp_lat', 'ipp_lon'.

    Returns
    -------
    np.ndarray
        Interpolated VTEC values (TECU).
    """

    # -------------------------
    # Prepare coordinates
    # -------------------------
    mapped_lon = (df["ipp_lon"] + 180) % 360 - 180
    lat = df["ipp_lat"].to_numpy()
    lon = mapped_lon.to_numpy()

    # -------------------------
    # Prepare epochs
    # -------------------------
    epochs = pd.to_datetime(df["obs_epoch"]).astype('datetime64[us]')
    gim_epochs = np.array(sorted(gim.keys()))

    # -------------------------
    # Build interpolators lazily
    # -------------------------
    interpolators = {
        t: RegularGridInterpolator(
            (lat_grid, lon_grid),
            gim[t],
            bounds_error=False,
            fill_value=np.nan,
            method="linear"
        )
        for t in gim_epochs
    }

    vtec_out = np.zeros(len(df))

    for i, t_obs in enumerate(epochs):

        # ---- find bracketing epochs ----
        idx = np.searchsorted(gim_epochs, t_obs)
        t0 = gim_epochs[max(idx - 1, 0)]
        t1 = gim_epochs[min(idx, len(gim_epochs) - 1)]

        # ---- spatial interpolation ----
        p = np.array([[lat[i], lon[i]]])   # shape (1, 2)
        v0 = interpolators[t0](p)[0]
        v1 = interpolators[t1](p)[0]

        # ---- temporal interpolation ----
        if t0 == t1:
            vtec_out[i] = v0
        else:
            w = (t_obs - t0) / (t1 - t0)
            vtec_out[i] = v0 + w * (v1 - v0)

    return vtec_out