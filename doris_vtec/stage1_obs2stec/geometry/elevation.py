import numpy as np
import pandas as pd
from pyproj import Transformer


def compute_pco_projection_cosines(df_obs: pd.DataFrame):
    """Return nominal satellite-axis and ground-Up LOS projection cosines.

    The LOS points from the station to the satellite. The positive satellite
    factor is the outward WGS84 ellipsoid normal dotted with that LOS,
    equivalently minus the nominal nadir antenna axis dotted with the LOS.
    The ground factor is local Up dotted with the same LOS, or sin(elevation).
    These are different axes and must not share one elevation factor.
    """
    station = df_obs[["sta_x", "sta_y", "sta_z"]].to_numpy(dtype=float)
    satellite = df_obs[["sat_x", "sat_y", "sat_z"]].to_numpy(dtype=float)
    los = satellite - station
    distance = np.linalg.norm(los, axis=1)
    valid = (
        np.isfinite(station).all(axis=1)
        & np.isfinite(satellite).all(axis=1)
        & (distance > 0.0)
        & (np.linalg.norm(station, axis=1) > 0.0)
        & (np.linalg.norm(satellite, axis=1) > 0.0)
    )
    satellite_cosine = np.full(len(df_obs), np.nan)
    ground_cosine = np.full(len(df_obs), np.nan)
    if not valid.any():
        return satellite_cosine, ground_cosine
    transformer = Transformer.from_crs(
        "EPSG:4978", "EPSG:4326", always_xy=True
    )

    def outward_normal(positions):
        lon, lat, _ = transformer.transform(
            positions[:, 0], positions[:, 1], positions[:, 2], radians=True
        )
        return np.column_stack((
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ))

    los_unit = los[valid] / distance[valid, None]
    satellite_cosine[valid] = np.einsum("ij,ij->i", outward_normal(satellite[valid]), los_unit)
    ground_cosine[valid] = np.einsum("ij,ij->i", outward_normal(station[valid]), los_unit)
    return satellite_cosine, ground_cosine


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
