import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tools import get_igs_vtec, haversine_vec, idw, get_gim_filename, latlon_to_unitvec
from readFile import read_ionFile
from pathlib import Path
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
import constant as const
import time

# === Configuration ===
satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]

def _split_df(df, n_parts):
    if len(df) == 0:
        return [df]
    return [df.iloc[s:e] for s, e in zip(
        np.linspace(0, len(df), n_parts + 1, dtype=int)[:-1],
        np.linspace(0, len(df), n_parts + 1, dtype=int)[1:]
    ) if e > s]

# --- NumPy-only ROTI --- change from vtec to stec needed
def compute_roti(epoch, vtec, station_code, pass_id, ref_epoch, time_gap_sec=150):
    """ROTI = std of dVTEC/dt (per minute) within ±time_gap_sec around ref_epoch."""

    in_win = np.abs((epoch - ref_epoch) / np.timedelta64(1, 's')) < float(time_gap_sec)
    if np.count_nonzero(in_win) < 2:
        return np.nan

    e  = epoch[in_win]
    v  = vtec[in_win]
    st = station_code[in_win]
    ps = pass_id[in_win]

    df_pairs = pd.DataFrame({'st': st, 'ps': ps})
    _, inv = np.unique(df_pairs[['st', 'ps']].apply(tuple, axis=1), return_inverse=True)

    order = np.lexsort((e, inv))
    e = e[order]; v = v[order]; inv = inv[order]

    same_group = inv[1:] == inv[:-1]
    if not np.any(same_group):
        return np.nan

    dt_min = (e[1:] - e[:-1]) / np.timedelta64(1, 'm')
    dvtec  = v[1:] - v[:-1]

    dt_min = dt_min.astype(float)
    valid = same_group & np.isfinite(dt_min) & (dt_min > 0)
    if not np.any(valid):
        return np.nan

    rates = dvtec[valid] / dt_min[valid]
    rates = rates[np.isfinite(rates)]
    return np.std(rates) if rates.size else np.nan

# --- Main routine ---
def altimetry_inpo(df_altimetry, df_doris, settings):
    
    df_doris = df_doris[df_doris['elevation'] > settings['ele_mask']].reset_index(drop=True)

    # --- Prepare output columns ---
    df_altimetry = df_altimetry.copy()
    df_altimetry['doris_vtec'] = np.nan
    df_altimetry['doris_points_count'] = 0
    df_altimetry['doris_points_std'] = np.nan
    df_altimetry['window_size'] = np.nan
    df_altimetry['roti'] = np.nan

    # --- Constants ---
    OMEGA = const.omega
    SEC = np.timedelta64(1, 's')
    min_distance = float(settings['min_distance'])
    max_distance = float(settings['max_distance'])
    min_obs = int(settings['min_obs_count'])

    # --- Per altimetry IPP ---
    for i, alt_row in df_altimetry.iterrows():
        lat0, lon0, t0 = alt_row['ipp_lat'], alt_row['ipp_lon'] % 360.0, alt_row['obs_epoch']

        # --- Time filter (|Δt| < 400 s) ---
        dt_sec = ((df_doris['obs_epoch'] - t0) / SEC).abs().astype(float)
        df_time = df_doris.loc[dt_sec < 400.0].reset_index(drop=True)
        if len(df_time) < min_obs:
            continue

        # --- Earth rotation correction ---
        delta_t = (df_time['obs_epoch'] - t0) / SEC
        dlon_deg = np.degrees(OMEGA * delta_t)
        df_time = df_time.assign(ipp_lon_corr=(df_time['ipp_lon'] + dlon_deg) % 360.0)

        # --- Compute ROTI ---
        roti_val = compute_roti(
            epoch=df_time['obs_epoch'],
            vtec=df_time['VTEC'],
            station_code=df_time['station_code'],
            pass_id=df_time['pass_id'],
            ref_epoch=t0
        )
        df_altimetry.at[i, 'roti'] = roti_val

        # --- KDTree search in (lat, lon) space converted to unit vectors ---
        src_vecs = latlon_to_unitvec(df_time['ipp_lat'], df_time['ipp_lon_corr'])
        target_vec = latlon_to_unitvec([lat0], [lon0])[0]
        tree = cKDTree(src_vecs)

        found = False
        cnt = 0
        cur_distance = min_distance

        while cur_distance <= max_distance:
            # Query neighbors within current radius (converted to unit-sphere chord)
            cand_idx = tree.query_ball_point(target_vec, r=cur_distance / const.AE84 * 1000)
            cnt = len(cand_idx)

            if cnt >= min_obs:
                df_neighbors = df_time.iloc[cand_idx]
                d_km = haversine_vec(lat0, lon0, df_neighbors['ipp_lat'], df_neighbors['ipp_lon_corr'])
                vtec_interp = idw(d_km, df_neighbors['VTEC'])

                df_altimetry.at[i, 'doris_vtec'] = vtec_interp
                df_altimetry.at[i, 'doris_points_std'] = np.nanstd(df_neighbors['VTEC'])
                df_altimetry.at[i, 'window_size'] = cur_distance
                found = True
                break

            cur_distance += 100.0  # km step

        df_altimetry.at[i, 'doris_points_count'] = cnt if found else 0

    return df_altimetry

def altimetry_distance_mask(df_altimetry, sta_xyz, max_km):
    R = const.AE84 
    lat = np.radians(df_altimetry['ipp_lat'].to_numpy())
    lon = np.radians(df_altimetry['ipp_lon'].to_numpy())
    ipp_xyz = np.column_stack([
        R * np.cos(lat) * np.cos(lon),
        R * np.cos(lat) * np.sin(lon),
        R * np.sin(lat)
    ])
    tree = cKDTree(sta_xyz)
    d, _ = tree.query(ipp_xyz, k=2)
    mask = ((d[:,0]+d[:,1])<= max_km*2*1000)
    return df_altimetry.loc[mask].copy()

# Main processing
if __name__ == '__main__':
    start_time = time.time()

    gim_dir = Path("/Volumes/T7Shield/DORISData/DORISInput/IGSGIM")
    doris_vtec_dir = Path("/Volumes/T7Shield/DORISData/DORISVTECStorage")
    alt_vtec_dir = Path("/Volumes/T7Shield/AltimetryData/AltimetryVTECStorage")
    dbav_dir = Path("/Volumes/T7Shield/DORISData/DBAVStorage")
    station_dir = Path("/Volumes/T7Shield/DORISData/StationCoord")

    proc_sate = satellite_list[1]
    range_ratio = range_ratio_list[0]
    mean_dist_stations = 1800  # km

    start_date = datetime(2024, 1, 1)
    proc_days = 1
    settings = {
        'ele_mask': 10,
        'min_obs_count': 30,
        'min_distance': 300,    # km
        'max_distance': 1000,   # km
    }

    for day_offset in range(proc_days):
        date = start_date + timedelta(days=day_offset)
        year, month, day = date.year, date.month, date.day
        doy = date.timetuple().tm_yday

        df_doris = pd.read_pickle(doris_vtec_dir / f"{proc_sate}/{year}/DOY{doy:03d}.pkl")
        df_altimetry = pd.read_pickle(alt_vtec_dir / f"{proc_sate}/{year}/DOY{doy:03d}.pkl")
        df_station = pd.read_pickle(station_dir / f"{year}" / f"DOY{doy:03d}.pkl")
        mask = df_station['station_code'].isin(df_doris['station_code'])
        df_matched = df_station.loc[mask, ['station_code', 'sta_x', 'sta_y', 'sta_z']].reset_index(drop=True)
        df_altimetry = altimetry_distance_mask(df_altimetry, df_matched[['sta_x', 'sta_y', 'sta_z']], mean_dist_stations)

        # --- GIM VTEC computation & comparison with Altimetry VTEC ---
        ion_file = get_gim_filename(gim_dir, year, doy)
        gim_vtec_raw = read_ionFile(str(ion_file))
        gim_vtec_scaled = gim_vtec_raw[:, ::-1, :] * 0.1
        gim_vtec_interp = get_igs_vtec(gim_vtec_scaled, df_altimetry)
        gim_vtec_diff = df_altimetry['VTEC'] - gim_vtec_interp * range_ratio
        gim_vtec_std = np.std(gim_vtec_diff)
        gim_vtec_rms = np.sqrt(np.mean(gim_vtec_diff ** 2))
        print(f"GIM VTEC - Std: {gim_vtec_std:.2f}, RMS: {gim_vtec_rms:.2f}")

        # --- DORIS VTEC interpolation (parallel processing) ---
        df_altimetry = df_altimetry.copy()
        df_altimetry['_orig_idx'] = df_altimetry.index

        n_parts = 10
        chunks = _split_df(df_altimetry, n_parts)
        results = []
        with ProcessPoolExecutor(max_workers=n_parts) as executor:
            futures = [executor.submit(altimetry_inpo, chunk, df_doris, settings) for chunk in chunks]
            for future in as_completed(futures):
                results.append(future.result())
        df_doris_result = pd.concat(results, axis=0)
        df_doris_result = df_doris_result.sort_values('_orig_idx').drop(columns=['_orig_idx'])

        # --- Comparison ---
        doris_vtec_diff = df_altimetry['VTEC'] - df_doris_result['doris_vtec']
        percent = np.count_nonzero(~np.isnan(doris_vtec_diff)) / len(doris_vtec_diff)
        doris_vtec_std = np.std(doris_vtec_diff)
        doris_vtec_rms = np.sqrt(np.mean(doris_vtec_diff ** 2))
        print(f"DORIS VTEC - Percent: {percent:.2%}, Std: {doris_vtec_std:.2f}, RMS: {doris_vtec_rms:.2f}")
        df_doris_result['gim_vtec'] = gim_vtec_interp
        df_doris_result.to_pickle(dbav_dir / f"{proc_sate}/{year}/DOY{doy}Ele{settings['ele_mask']}.pkl")

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
