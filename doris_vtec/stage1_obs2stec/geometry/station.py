import pandas as pd


SECONDS_PER_YEAR = 365.25 * 86400.0

def add_station_position(
    df_obs: pd.DataFrame,
    df_sta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add station ECEF coordinates to observation DataFrame.

    For each station, a single station position is computed using the
    first observation epoch of that station and applied to all
    observations of the same station.

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
    sta_records = []

    # ------------------------------------------
    # Compute one station position per station
    # ------------------------------------------
    for station_code, obs_group in df_obs.groupby("station_code"):
        # Representative epoch: first epoch of the raw observations
        t_ref = obs_group["obs_epoch"].iloc[0]

        sta_group = df_sta[df_sta["station_code"] == station_code]

        valid = (
            (sta_group["start_obs"] <= t_ref)
            &  (t_ref <= sta_group["end_obs"])
        )

        if valid.sum() != 1:
            continue  # no valid or ambiguous solution

        sol = sta_group.loc[valid].iloc[0]

        dt_year = (
            (t_ref - sol["ref_epoch"]).total_seconds()
            / SECONDS_PER_YEAR
        )

        sta_records.append(
            {
                "station_code": station_code,
                "antenna_type": sol["antenna_type"],
                "sta_x": sol["x"] + sol["vx"] * dt_year,
                "sta_y": sol["y"] + sol["vy"] * dt_year,
                "sta_z": sol["z"] + sol["vz"] * dt_year,
            }
        )

    df_sta_day = pd.DataFrame(sta_records)

    # ------------------------------------------
    # Attach station positions to observations
    # ------------------------------------------
    return df_obs.merge(df_sta_day, on="station_code", how="left")