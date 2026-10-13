from pathlib import Path
import numpy as np
import pandas as pd

from ..io.ionex_reader import read_ionex_file
from ..io.utils import (
    find_covering_gim_file_from_dir, correct_receiver_epochs, tai_epochs_to_utc,
)

from .pass_detection.detect_passes import detect_passes
from .leveling.reference_point import select_max_elevation_points
from .leveling.gim_interpolation import get_gim_vtec
from .leveling.apply_leveling import apply_passwise_leveling


def process_day(
    date: pd.Timestamp,
    obs_root: Path,
    gim_root: Path,
    out_root: Path,
    satellite: str,
    config: dict,
) -> pd.DataFrame:
    """
    Process one day of DORIS observations:
    RINEX (STEC) -> pass detection -> GIM leveling -> VTEC.
    """

    year = date.year
    doy = date.dayofyear

    obs_path = (obs_root/f"{year}"/f"DOY{doy:03d}.pkl")
    df_obs = pd.read_pickle(obs_path)

    if df_obs is None or df_obs.empty:
        return pd.DataFrame()

    if (
        df_obs.attrs.get("pco_projection_model") != "nominal_los_v1"
        or df_obs.attrs.get("obs_epoch_receiver_clock_corrected") is not True
        or df_obs.attrs.get("obs_epoch_time_scale") != "TAI"
    ):
        raise ValueError("Rebuild Stage1 with the clock and LOS-PCO fixes before Stage2.")
    # Preserve the corrected epoch; do not add the receiver offset twice.
    df_obs = correct_receiver_epochs(df_obs)

    # -------------------------------------------------
    # Pass detection (station-wise)
    # -------------------------------------------------
    pass_dfs = []

    for _, df_sta in df_obs.groupby("station_code"):
        df_pass = detect_passes(
            obs_per_station=df_sta,
            min_obs_count=config["min_obs_count"],
            elev_thres=config["elev_thres"],
        )
        if not df_pass.empty:
            pass_dfs.append(df_pass)

    if not pass_dfs:
        return pd.DataFrame()

    df_pass = pd.concat(pass_dfs, ignore_index=True)

    # -------------------------------------------------
    # Reference point & GIM VTEC
    # -------------------------------------------------
    df_ref = select_max_elevation_points(df_pass)

    df_ref = df_ref.copy()
    df_ref["gim_vtec"] = np.nan
    utc_days = pd.DatetimeIndex(tai_epochs_to_utc(df_ref["obs_epoch"])).normalize()
    for utc_day in utc_days.unique():
        gim_path = find_covering_gim_file_from_dir(
            gim_dir=gim_root, year=utc_day.year, doy=utc_day.dayofyear,
        )
        if gim_path is None:
            raise FileNotFoundError(f"GIM missing for required UTC day {utc_day.date()}")
        gim, lat_grid, lon_grid = read_ionex_file(gim_path)
        selected = utc_days == utc_day
        df_ref.loc[selected, "gim_vtec"] = get_gim_vtec(
            gim=gim, lat_grid=lat_grid, lon_grid=lon_grid,
            df=df_ref.loc[selected], observation_time_system="TAI",
        )

    # -------------------------------------------------
    # Pass-wise leveling
    # -------------------------------------------------
    
    df_vtec = apply_passwise_leveling(
        df_pass=df_pass,
        df_ref=df_ref,
        range_ratio=config[f"range_ratio_{satellite}"],
        min_ref_elev=config["min_ref_elev"],
    )

    if df_vtec.empty:
        return pd.DataFrame()
        
    columns_to_keep = [
        'obs_epoch', 'ipp_lat', 'ipp_lon',
        'stec', 'vtec', 'elevation', 'map_value',
        'station_code', 'prn', 'pass_id'
    ]
    df_vtec = df_vtec[columns_to_keep].dropna(subset=["vtec"])
    df_vtec.attrs.update(obs_epoch_time_scale="TAI", obs_epoch_receiver_clock_corrected=True,
                         pco_projection_model="nominal_los_v1")
    # -------------------------------------------------
    # Write output
    # -------------------------------------------------
    out_path = (
        out_root
        / f"{year}"
        / f"DOY{doy:03d}.pickle"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_vtec.to_pickle(out_path)
