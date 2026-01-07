import numpy as np
import pandas as pd
from pyproj import Transformer


def add_elevation(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Add elevation angle to observation DataFrame.

    Elevation is computed in the local ENU frame at the station.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Must contain columns:
        - sta_x, sta_y, sta_z   (meters, ECEF)
        - sat_x, sat_y, sat_z   (meters, ECEF)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional column:
        - elevation   (degrees)
    """
    # ------------------------------------------
    # ECEF -> geodetic (station position)
    # ------------------------------------------
    transformer = Transformer.from_crs(
        "EPSG:4978", "EPSG:4326", always_xy=True
    )

    lon, lat, _ = transformer.transform(
        df_obs["sta_x"].values,
        df_obs["sta_y"].values,
        df_obs["sta_z"].values,
        radians=True,
    )

    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)

    # ------------------------------------------
    # ECEF -> ENU rotation matrix
    # ------------------------------------------
    R = np.array([
        [-sin_lon,              cos_lon,               np.zeros_like(lon)],
        [-sin_lat * cos_lon, -sin_lat * sin_lon,  cos_lat],
        [ cos_lat * cos_lon,  cos_lat * sin_lon,  sin_lat],
    ]).transpose(2, 0, 1)  # (N, 3, 3)

    # ------------------------------------------
    # Line-of-sight vector in ENU
    # ------------------------------------------
    vec = (
        df_obs[["sat_x", "sat_y", "sat_z"]].values
        - df_obs[["sta_x", "sta_y", "sta_z"]].values
    )

    enu = np.einsum("nij,nj->ni", R, vec)

    # ------------------------------------------
    # Elevation angle
    # ------------------------------------------
    elevation_rad = np.arctan2(
        enu[:, 2],
        np.linalg.norm(enu[:, :2], axis=1),
    )

    df_obs["elevation"] = np.rad2deg(elevation_rad)

    return df_obs