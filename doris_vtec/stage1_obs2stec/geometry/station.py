import numpy as np
import pandas as pd
from ...io.utils import tai_epochs_to_utc


SECONDS_PER_YEAR = 365.25 * 86400.0

def add_station_position(
    df_obs: pd.DataFrame,
    df_sta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add station ECEF coordinates to observation DataFrame.

    For each matched SINEX solution interval, a single station position is
    computed using its first observation epoch and applied to observations
    in that interval. Unmatched rows retain missing coordinates.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Observation DataFrame containing:
        - station_code
        - obs_epoch

    df_sta : pd.DataFrame
        Station solution DataFrame containing:
        - station_code
        - antenna_type
        - start_obs
        - end_obs
        - ref_epoch
        - x, y, z        (meters)
        - vx, vy, vz     (meters / year)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - sta_x, sta_y, sta_z   (meters)
    """
    result = df_obs.copy().reset_index(drop=True)
    result["antenna_type"] = ""
    result[["sta_x", "sta_y", "sta_z"]] = np.nan
    assigned = pd.Series(False, index=result.index)

    # ------------------------------------------
    # Compute one station position per matched solution interval
    # ------------------------------------------
    for station_code, obs_group in result.groupby("station_code"):
        times = obs_group["obs_epoch"].copy()
        if df_obs.attrs.get("obs_epoch_receiver_clock_corrected") is True:
            # SINEX validity limits and reference epochs are UTC labels.
            times = pd.Series(
                tai_epochs_to_utc(times.to_numpy()), index=obs_group.index
            )

        sta_group = df_sta[df_sta["station_code"] == station_code]
        for sol in sta_group.itertuples(index=False):
            valid = times.between(sol.start_obs, sol.end_obs, inclusive="both")
            idx = times.index[valid]
            if len(idx) == 0:
                continue
            if assigned.loc[idx].any():
                raise ValueError(f"Overlapping SINEX solutions: {station_code}")

            # Retain the old representative-epoch position within each interval.
            t_ref = times.loc[idx].iloc[0]
            dt_year = (t_ref - sol.ref_epoch).total_seconds() / SECONDS_PER_YEAR

            for axis in "xyz":
                result.loc[idx, f"sta_{axis}"] = (
                    getattr(sol, axis) + getattr(sol, f"v{axis}") * dt_year
                )
            result.loc[idx, "antenna_type"] = sol.antenna_type
            assigned.loc[idx] = True

    result.attrs.update(df_obs.attrs)
    return result
