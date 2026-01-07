import numpy as np
import constant
from ..geo.spherical import haversine_vec, latlon_to_unitvec
from .weights import idw
from scipy.spatial import cKDTree
import warnings



def filter_doris_in_time(df_doris, t0, time_window_sec=400.0):
    """
    Select DORIS observations within a symmetric time window
    around a reference epoch.
    """
    SEC = np.timedelta64(1, 's')
    dt_sec = ((df_doris['obs_epoch'] - t0) / SEC).abs().astype(float)
    return df_doris.loc[dt_sec < time_window_sec].reset_index(drop=True)


def apply_earth_rotation(df_time, t0):
    """
    Apply Earth rotation correction to DORIS IPP longitudes
    relative to a reference epoch.
    """
    SEC = np.timedelta64(1, 's')
    OMEGA = constant.omega

    delta_t = (df_time['obs_epoch'] - t0) / SEC
    dlon_deg = np.degrees(OMEGA * delta_t)

    df_out = df_time.copy()
    df_out['ipp_lon_corr'] = (df_out['ipp_lon'] + dlon_deg) % 360.0
    return df_out


def interpolate_single_ipp(
    lat0, lon0,
    df_doris,
    settings
):
    """
    Interpolate DORIS VTEC to a single target IPP
    using adaptive-radius IDW.
    """

    # Build KDTree in unit-vector space
    src_vecs = latlon_to_unitvec(
        df_doris['ipp_lat'],
        df_doris['ipp_lon_corr']
    )
    target_vec = latlon_to_unitvec([lat0], [lon0])[0]
    tree = cKDTree(src_vecs)

    cur_distance = settings["min_distance"]

    while cur_distance <= settings["max_distance"]:
        r_chord = cur_distance / constant.AE84 * 1000.0
        cand_idx = tree.query_ball_point(target_vec, r=r_chord)

        if len(cand_idx) >= settings["min_obs_count"]:
            df_sel = df_doris.iloc[cand_idx]

            d_km = haversine_vec(
                lat0, lon0,
                df_sel['ipp_lat'],
                df_sel['ipp_lon_corr']
            )

            vtec = idw(d_km, df_sel['vtec'])

            std = np.nanstd(df_sel["vtec"])


            return {
                "vtec": vtec,
                "count": len(df_sel),
                "std": std,
                "window": cur_distance
            }

        cur_distance += settings["distance_step"]

    return None


def altimetry_inpo(df_altimetry, df_doris, settings):
    """
    Interpolate DORIS-derived VTEC to altimetry IPPs.
    """

    # Elevation mask
    df_doris = df_doris[df_doris['elevation'] > settings["elev_thres"]]
    df_doris = df_doris.reset_index(drop=True)

    n_alt = len(df_altimetry)

    vtec_arr   = np.full(n_alt, np.nan)
    cnt_arr    = np.zeros(n_alt, dtype=int)
    std_arr    = np.full(n_alt, np.nan)
    win_arr    = np.full(n_alt, np.nan)

    for i, alt in df_altimetry.reset_index(drop=True).iterrows():
        t0   = alt['obs_epoch']
        lat0 = alt['ipp_lat']
        lon0 = alt['ipp_lon'] % 360.0

        df_time = filter_doris_in_time(df_doris, t0)
        if len(df_time) < settings["min_obs_count"]:
            continue

        df_prepared = apply_earth_rotation(df_time, t0)

        result = interpolate_single_ipp(
            lat0, lon0,
            df_prepared,
            settings
        )

        if result is None:
            continue

        vtec_arr[i] = result["vtec"]
        cnt_arr[i]  = result["count"]
        std_arr[i]  = result["std"]
        win_arr[i]  = result["window"]

    df_out = df_altimetry.copy()
    df_out['doris_vtec_idw']     = vtec_arr
    df_out['doris_points_count'] = cnt_arr
    df_out['doris_points_std']   = std_arr
    df_out['window_size']        = win_arr

    return df_out