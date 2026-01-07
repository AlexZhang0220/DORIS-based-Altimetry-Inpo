from pathlib import Path
import pandas as pd

from ..io.ionex_reader import read_ionex_file
from ..io.utils import find_covering_gim_file_from_dir

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

    # -------------------------------------------------
    # Find and read covering GIM file
    # -------------------------------------------------
    gim_path = find_covering_gim_file_from_dir(
        gim_dir=gim_root,
        year=year,
        doy=doy,
    )

    if gim_path is None:
        return pd.DataFrame()

    gim, lat_grid, lon_grid = read_ionex_file(gim_path)

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
    df_ref["gim_vtec"] = get_gim_vtec(
        gim=gim,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        df=df_ref,
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
