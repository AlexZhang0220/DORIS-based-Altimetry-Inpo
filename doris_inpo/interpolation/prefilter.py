import numpy as np
import constant
from scipy.spatial import cKDTree


def altimetry_distance_mask(df_altimetry, sta_xyz, max_km):
    """
    Geometric pre-filter for altimetry IPPs.

    Altimetry IPPs are retained only if they are sufficiently close to the
    DORIS station network. The criterion is based on the combined distance
    to the two nearest stations, which reduces the impact of isolated
    single-station geometry.

    Parameters
    ----------
    df_altimetry : pandas.DataFrame
        Altimetry data with IPP latitude and longitude.
    sta_xyz : ndarray
        Cartesian coordinates of DORIS stations.
    max_km : float
        Distance threshold in kilometers.

    Returns
    -------
    pandas.DataFrame
        Filtered altimetry data.
    """
    R = constant.AE84 
    lat = np.radians(df_altimetry['ipp_lat'].to_numpy())
    lon = np.radians(df_altimetry['ipp_lon'].to_numpy())
    ipp_xyz = np.column_stack([
        R * np.cos(lat) * np.cos(lon),
        R * np.cos(lat) * np.sin(lon),
        R * np.sin(lat)
    ])
    tree = cKDTree(sta_xyz)
    d, _ = tree.query(ipp_xyz, k=2)
    mask = (d[:,1]+d[:,0])<= max_km*1000*2
    return df_altimetry.loc[mask].copy()