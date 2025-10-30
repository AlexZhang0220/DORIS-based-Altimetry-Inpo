import numpy as np
import pandas as pd
import os
import re
from datetime import datetime, timedelta
from tools import get_igs_vtec, haversine_vec, idw, get_gim_filename, latlon_to_unitvec
from readFile import read_ionFile
from pathlib import Path
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
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

    e  = epoch[in_win].to_numpy()
    v  = vtec[in_win].to_numpy()
    st = station_code[in_win].to_numpy()
    ps = pass_id[in_win].to_numpy()

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
        if settings['enable_roti_flag']:
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

            cur_distance += settings['distance_step']  # km 

        df_altimetry.at[i, 'doris_points_count'] = cnt if found else 0

    return df_altimetry

# --- Result storage and comparison ---
def doris_result_comparison(year, doris_result_dir, altimetry_vtec_dir, comparison_dir):

    doris_path = Path(doris_result_dir/str(year))
    alt_path = Path(altimetry_vtec_dir/str(year))

    def extract_doy_key(name: str):
        m = re.search(r'DOY(\d{3})', name)
        return f"DOY{m.group(1)}" if m else None

    results = []
    doris_files = sorted(
        f for f in doris_path.glob("*.pkl")
        if not f.name.startswith("._")
    )

    for f in doris_files:
        doy_key = extract_doy_key(f.name)
        df_doris = pd.read_pickle(f)
        if not all(col in df_doris.columns for col in ["VTEC", "doris_vtec", "gim_vtec"]):
            continue

        # Compute stats
        diff_doris = df_doris["VTEC"] - df_doris["doris_vtec"]
        diff_gim   = df_doris["VTEC"] - df_doris["gim_vtec"] * range_ratio_list[0]

        rms_doris = float(np.sqrt(np.nanmean(np.square(diff_doris)))) if len(diff_doris) else np.nan
        std_doris = float(np.nanstd(diff_doris)) if len(diff_doris) else np.nan
        rms_gim   = float(np.sqrt(np.nanmean(np.square(diff_gim)))) if len(diff_gim) else np.nan
        std_gim   = float(np.nanstd(diff_gim)) if len(diff_gim) else np.nan

        # Find altimetry file
        alt_files = list(f for f in alt_path.glob(f"*{doy_key}*.pkl") if not f.name.startswith("._"))
        df_alt = pd.read_pickle(alt_files[0])
        n_alt = len(df_alt)
        n_doris = len(df_doris)
        percentage = n_doris / n_alt if n_alt and not np.isnan(n_alt) else np.nan

        results.append({
            "doy": doy_key,
            "n_doris": n_doris,
            "n_alt": n_alt,
            "percentage": percentage,
            "rms_doris": rms_doris,
            "std_doris": std_doris,
            "rms_gim": rms_gim,
            "std_gim": std_gim
        })

    summary = pd.DataFrame(results).sort_values("doy").reset_index(drop=True)
    summary.to_csv(comparison_dir / f"{year}_comparison_summary.csv", index=False, float_format="%.4f")
    print(f"Comparison summary saved to {comparison_dir / f'{year}_comparison_summary.csv'}")

# Main processing
if __name__ == '__main__':
    start_time = time.time()

    gim_dir = Path("/Volumes/T7Shield/DORISData/DORISInput/IGSGIM")
    doris_vtec_dir = Path("/Volumes/T7Shield/DORISData/DORISVTECStorage")
    alt_vtec_dir = Path("/Volumes/T7Shield/AltimetryData/AltimetryVTECStorage")
    dbav_dir = Path("/Volumes/T7Shield/DORISData/DBAVStorage")
    station_dir = Path("/Volumes/T7Shield/DORISData/StationCoord")
    result_dir = Path("/Volumes/T7Shield/DORISData/ComparisonResults")

    proc_sate = satellite_list[1]
    range_ratio = range_ratio_list[0]
    mean_dist_stations = 1700   # km

    start_date = datetime(2016, 3, 12)
    proc_days = 1
    settings = {
        'ele_mask': 10,
        'min_obs_count': 30,
        'min_distance': 800,    # km
        'max_distance': 1000,   # km
        'distance_step': 200,   # km
        'enable_roti_flag': False
    }

    pbar = tqdm(range(proc_days), desc="DORIS Interpolation", unit="day")
    for day_offset in pbar:
        date = start_date + timedelta(days=day_offset)
        year, month, day = date.year, date.month, date.day
        doy = date.timetuple().tm_yday

        doris_file = Path(doris_vtec_dir) / f"{proc_sate}/{year}/DOY{doy:03d}.pkl"
        alt_file = Path(alt_vtec_dir) / f"{proc_sate}/{year}/DOY{doy:03d}.pkl"
        sta_file = Path(station_dir) / f"{year}/DOY{doy:03d}.pkl"

        if not (doris_file.exists() and alt_file.exists() and sta_file.exists()):
            print(f"Skip DOY{doy:03d} — file not found")
            continue

        # Safely read pickle files
        df_doris = pd.read_pickle(doris_file)
        df_altimetry = pd.read_pickle(alt_file)
        df_station = pd.read_pickle(sta_file)
        mask = df_station['station_code'].isin(df_doris['station_code'])
        df_matched = df_station.loc[mask, ['station_code', 'sta_x', 'sta_y', 'sta_z']].reset_index(drop=True)

        start = pd.Timestamp('2016-03-12 22:51:41')
        end   = pd.Timestamp('2016-03-12 23:47:53')

        df_altimetry_masked = df_altimetry[
            (df_altimetry['obs_epoch'] >= start) &
            (df_altimetry['obs_epoch'] <= end)
        ]
        # df_altimetry_masked = altimetry_distance_mask(df_altimetry, df_matched[['sta_x', 'sta_y', 'sta_z']], mean_dist_stations)

        # --- GIM VTEC computation & comparison with Altimetry VTEC ---
        ion_file = get_gim_filename(gim_dir, year, doy)
        gim_vtec_raw = read_ionFile(str(ion_file))
        gim_vtec = gim_vtec_raw[:, ::-1, :] * 0.1
        gim_vtec_interp = get_igs_vtec(gim_vtec, df_altimetry_masked) # gim data integrated after doris processing

        # --- DORIS VTEC interpolation (parallel processing) ---
        df_altimetry_masked = df_altimetry_masked.copy()
        df_altimetry_masked['_orig_idx'] = df_altimetry_masked.index

        n_parts = 10
        chunks = _split_df(df_altimetry_masked, n_parts)
        results = []
        with ProcessPoolExecutor(max_workers=n_parts) as executor:
            futures = [executor.submit(altimetry_inpo, chunk, df_doris, settings) for chunk in chunks]
            for future in as_completed(futures):
                results.append(future.result())
        df_doris_result = pd.concat(results, axis=0)
        df_doris_result = df_doris_result.sort_values('_orig_idx').drop(columns=['_orig_idx'])
        df_doris_result['gim_vtec'] = gim_vtec_interp

        save_dir = dbav_dir / f"{proc_sate}/{mean_dist_stations}km/{year}"
        os.makedirs(save_dir, exist_ok=True)
        df_doris_result.to_pickle(save_dir / f"DOY{doy:03d}.pkl")

    # --- Result comparison ---
    doris_result_comparison(year, dbav_dir/proc_sate/f'{mean_dist_stations}km', alt_vtec_dir/proc_sate, result_dir)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
