import numpy as np
import pandas as pd
from pyproj import Transformer
import constant


def add_ipp_position(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Add ionospheric pierce point (IPP) coordinates to observation DataFrame.

    IPP is computed as the intersection of the station-to-satellite line
    with a single-layer ionospheric shell.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Must contain columns:
        - sta_x, sta_y, sta_z   (meters, ECEF)
        - sat_x, sat_y, sat_z   (meters, ECEF)
        - elevation             (degrees)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - ipp_lon   (degrees)
        - ipp_lat   (degrees)
    """
    AE84 = constant.AE84
    ion_height = constant.ION_HEIGHT

    # --------------------------------------------------
    # Station and satellite ECEF positions
    # --------------------------------------------------
    sta = df_obs[["sta_x", "sta_y", "sta_z"]].values
    sat = df_obs[["sat_x", "sat_y", "sat_z"]].values

    # --------------------------------------------------
    # Elevation-related geometry
    # --------------------------------------------------
    alpha = np.deg2rad(90.0 + df_obs["elevation"].values)

    # --------------------------------------------------
    # Station geodetic height
    # --------------------------------------------------
    transformer = Transformer.from_crs(
        "EPSG:4978", "EPSG:4326", always_xy=True
    )
    _, _, sta_h = transformer.transform(
        sta[:, 0], sta[:, 1], sta[:, 2]
    )

    # --------------------------------------------------
    # Line-of-sight geometry
    # --------------------------------------------------
    vec_rs = sat - sta
    D_rs = np.linalg.norm(vec_rs, axis=1)

    term1 = (AE84 + sta_h) * np.cos(alpha)
    under_sqrt = (
        term1**2
        + (AE84 + ion_height)**2
        - (AE84 + sta_h)**2
    )
    under_sqrt = np.maximum(under_sqrt, 0.0)

    term2 = np.sqrt(under_sqrt)

    D_rP1 = term1 + term2
    D_rP2 = term1 - term2

    scale1 = D_rP1 / D_rs
    scale2 = D_rP2 / D_rs

    vec1 = sta + vec_rs * scale1[:, None]
    vec2 = sta + vec_rs * scale2[:, None]

    # --------------------------------------------------
    # Choose the physically valid intersection
    # --------------------------------------------------
    dist1 = np.linalg.norm(sat - vec1, axis=1)
    dist2 = np.linalg.norm(sat - vec2, axis=1)

    ipp_ecef = np.where(
        (dist1 <= dist2)[:, None],
        vec1,
        vec2,
    )

    # --------------------------------------------------
    # Convert IPP to geographic coordinates
    # --------------------------------------------------
    ipp_lon, ipp_lat, _ = transformer.transform(
        ipp_ecef[:, 0],
        ipp_ecef[:, 1],
        ipp_ecef[:, 2],
    )

    df_obs["ipp_lon"] = ipp_lon
    df_obs["ipp_lat"] = ipp_lat

    return df_obs