import numpy as np
import pandas as pd
import os
import re
import tools
from pathlib import Path
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import constant as const
import CycleBasedRead
import time

def process_single_pass(pid: int,
                        df_alt_cycle: pd.DataFrame,
                        df_doris_cycle: pd.DataFrame,
                        df_station_cycle: pd.DataFrame,
                        gim_cycle: dict,
                        settings: dict,
                        cycle_number: int) -> pd.DataFrame | None:
    """
    Process a single pass in an independent process.

    Steps:
    1) Select altimetry observations belonging to this pass.
    2) Find corresponding DORIS observations using a time window.
    3) Match stations and apply distance-based filtering on altimetry IPPs.
    4) Compute GIM VTEC for this pass (supports cross-day passes).
    5) Call altimetry_inpo to perform DORIS interpolation.
    6) Add cycle_number, pass_in_cycle, and gim_vtec fields, then return the result.
    """

    df_alt_pass = df_alt_cycle[df_alt_cycle["pass_in_cycle"] == pid].copy()
    if df_alt_pass.empty:
        print(f"[PASS {pid}] no altimetry, skip.")
        return None

    pass_start = df_alt_pass["obs_epoch"].min()
    pass_end = df_alt_pass["obs_epoch"].max()

    win_start = pass_start - pd.Timedelta(minutes=5)
    win_end = pass_end + pd.Timedelta(minutes=5)

    # DORIS subset
    df_doris_pass = df_doris_cycle[
        (df_doris_cycle["obs_epoch"] >= win_start) &
        (df_doris_cycle["obs_epoch"] <= win_end)
    ].copy()
    if df_doris_pass.empty:
        print(f"[PASS {pid}] no DORIS data in window, skip.")
        return None

    # Station subset
    mask_sta = df_station_cycle["station_code"].isin(df_doris_pass["station_code"])
    df_sta_pass = df_station_cycle.loc[
        mask_sta, ['station_code', 'sta_x', 'sta_y', 'sta_z']
    ].reset_index(drop=True)
    if df_sta_pass.empty:
        print(f"[PASS {pid}] no station matched with DORIS, skip.")
        return None

    # Distance filtering altimetry
    df_alt_pass_masked = altimetry_distance_mask(
        df_alt_pass,
        df_sta_pass[['sta_x', 'sta_y', 'sta_z']],
        settings['distance_station']
    )
    if df_alt_pass_masked.empty:
        print(f"[PASS {pid}] no altimetry after distance mask, skip.")
        return None

    # Gim vtec subset
    df_alt_pass_masked = df_alt_pass_masked.copy()
    dates = df_alt_pass_masked["obs_epoch"].dt.normalize()
    unique_dates = dates.unique()

    gim_vtec_interp = np.full(len(df_alt_pass_masked), np.nan)

    for date_i in unique_dates:
        year_i = date_i.year
        doy_i = date_i.timetuple().tm_yday
        gim_vtec = gim_cycle[year_i, doy_i]

        mask_day = (dates == date_i)
        df_alt_sub = df_alt_pass_masked.loc[mask_day]

        gim_vtec_sub = tools.get_igs_vtec(gim_vtec, df_alt_sub)
        gim_vtec_interp[mask_day.to_numpy()] = gim_vtec_sub

    # --- interpolation process ---
    df_inpo = altimetry_inpo(df_alt_pass_masked, df_doris_pass, settings)
    df_inpo["cycle_number"] = cycle_number
    df_inpo["pass_in_cycle"] = pid
    df_inpo["gim_vtec"] = gim_vtec_interp

    return df_inpo

# === Configuration ===
satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]

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
    # second nearest station should be no further than max_km
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
    mask = (d[:,1]+d[:,0])<= max_km*1000*2
    return df_altimetry.loc[mask].copy()

# --- Main routine ---
def altimetry_inpo(df_altimetry: pd.DataFrame,
                   df_doris: pd.DataFrame,
                   settings: dict) -> pd.DataFrame:

    # --- Elevation mask for DORIS ---
    df_doris = df_doris[df_doris['elevation'] > settings['ele_mask']].reset_index(drop=True)

    # --- Constants ---
    OMEGA = const.omega
    SEC = np.timedelta64(1, 's')
    min_distance = float(settings['min_distance'])
    max_distance = float(settings['max_distance'])
    min_obs = int(settings['min_obs_count'])
    enable_roti = bool(settings.get('enable_roti_flag', False))

    n_alt = len(df_altimetry)
    doris_vtec_idw = np.full(n_alt, np.nan)
    doris_points_count = np.zeros(n_alt, dtype=int)
    doris_points_std = np.full(n_alt, np.nan)
    window_size = np.full(n_alt, np.nan)
    station_count = np.full(n_alt, np.nan)
    roti_arr = np.full(n_alt, np.nan)
    ipp_lat_str = pd.Series(index=df_altimetry.index, dtype="object")
    ipp_lon_str = pd.Series(index=df_altimetry.index, dtype="object")

    # --- Loop per altimetry IPP ---
    for idx_alt, alt_row in df_altimetry.reset_index(drop=True).iterrows():
        lat0 = alt_row['ipp_lat']
        lon0 = alt_row['ipp_lon'] % 360.0
        t0 = alt_row['obs_epoch']

        # --- Time filter (|Δt| < 400 s) ---
        dt_sec = ((df_doris['obs_epoch'] - t0) / SEC).abs().astype(float)
        df_time = df_doris.loc[dt_sec < 400.0].reset_index(drop=True)
        if len(df_time) < min_obs:
            continue

        # --- Earth rotation correction ---
        delta_t = (df_time['obs_epoch'] - t0) / SEC
        dlon_deg = np.degrees(OMEGA * delta_t)
        df_time = df_time.assign(ipp_lon_corr=(df_time['ipp_lon'] + dlon_deg) % 360.0)

        # --- Compute ROTI (optional) ---
        if enable_roti:
            roti_val = compute_roti(
                epoch=df_time['obs_epoch'],
                vtec=df_time['VTEC'],
                station_code=df_time['station_code'],
                pass_id=df_time['pass_id'],
                ref_epoch=t0
            )
            roti_arr[idx_alt] = roti_val

        # --- KDTree search in (lat, lon) space converted to unit vectors ---
        src_vecs = tools.latlon_to_unitvec(df_time['ipp_lat'], df_time['ipp_lon_corr'])
        target_vec = tools.latlon_to_unitvec([lat0], [lon0])[0]
        tree = cKDTree(src_vecs)

        cur_distance = min_distance

        while cur_distance <= max_distance:
            # Query neighbors within current radius (converted to unit-sphere chord)
            r_chord = cur_distance / const.AE84 * 1000.0
            cand_idx = tree.query_ball_point(target_vec, r=r_chord)
            cnt = len(cand_idx)

            if cnt >= min_obs:
                df_neighbors = df_time.iloc[cand_idx]
                d_km = tools.haversine_vec(
                    lat0, lon0,
                    df_neighbors['ipp_lat'],
                    df_neighbors['ipp_lon_corr']
                )

                doris_vtec_idw[idx_alt] = tools.idw(d_km, df_neighbors['VTEC'])
                doris_points_count[idx_alt] = cnt
                doris_points_std[idx_alt] = np.nanstd(df_neighbors['VTEC'])
                window_size[idx_alt] = cur_distance
                # station_count[idx_alt] = df_neighbors['station_code'].nunique()

                # ipp_lat_str.iloc[idx_alt] = ",".join(f"{v:.4f}" for v in df_neighbors['ipp_lat'].to_numpy())
                # ipp_lon_str.iloc[idx_alt] = ",".join(f"{v:.4f}" for v in df_neighbors['ipp_lon'].to_numpy())
                break

            cur_distance += settings['distance_step']  # km

    # --- Attach results to a copy of df_altimetry ---
    df_out = df_altimetry.copy()
    df_out['doris_vtec_idw'] = doris_vtec_idw
    df_out['doris_points_count'] = doris_points_count
    df_out['doris_points_std'] = doris_points_std
    df_out['window_size'] = window_size
    # df_out['station_count'] = station_count
    if enable_roti: df_out["roti"] = roti_arr
    # df_out['doris_ipp_lat'] = ipp_lat_str
    # df_out['doris_ipp_lon'] = ipp_lon_str

    return df_out

# --- Result storage and comparison ---
def doris_result_comparison(year, doris_result_dir, altimetry_vtec_dir, comparison_dir):

    doris_path = Path(doris_result_dir)
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
        diff_doris_idw = df_doris["VTEC"] - df_doris["doris_vtec_idw"]
        diff_gim   = df_doris["VTEC"] - df_doris["gim_vtec"] * range_ratio_list[0]

        rms_doris = float(np.sqrt(np.nanmean(np.square(diff_doris)))) if len(diff_doris) else np.nan
        std_doris = float(np.nanstd(diff_doris)) if len(diff_doris) else np.nan

        rms_doris_idw = float(np.sqrt(np.nanmean(np.square(diff_doris_idw)))) if len(diff_doris_idw) else np.nan
        std_doris_idw = float(np.nanstd(diff_doris_idw)) if len(diff_doris_idw) else np.nan

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
            "rms_doris_idw": rms_doris_idw,
            "std_doris_idw": std_doris_idw,
            "rms_gim": rms_gim,
            "std_gim": std_gim
        })

    summary = pd.DataFrame(results).sort_values("doy").reset_index(drop=True)
    summary.to_csv(comparison_dir / f"{year}_comparison_summary.csv", index=False, float_format="%.4f")
    print(f"Comparison summary saved to {comparison_dir / f'{year}_comparison_summary.csv'}")

# Main processing
if __name__ == '__main__':
    start_time = time.time()

    paths = {
        "gim": Path("/Volumes/T7Shield/DORISData/DORISInput/IGSGIM"),
        "doris_vtec": Path("/Volumes/T7Shield/DORISData/DORISVTECStorage"),
        "alt_vtec": Path("/Volumes/T7Shield/AltimetryData/AltimetryVTECStorage"),
        "dbav": Path("/Volumes/T7Shield/DORISData/DBAVStorage"),
        "station": Path("/Volumes/T7Shield/DORISData/StationCoord"),
        "result": Path("/Volumes/T7Shield/DORISData/ComparisonResults"),
    }

    proc_sate = satellite_list[1]
    range_ratio = range_ratio_list[0]

    cycle_list = list(range(201, 391, 1))

    settings = {
        'ele_mask': 10,
        'min_obs_count': 30,
        'min_distance': 100,      # km
        'max_distance': 1000,     # km
        'distance_step': 100,     # km
        'distance_station': 1675, # km 
        'enable_roti_flag': False 
    }

    # preload both cycle tables once
    cycle_table_f     = CycleBasedRead.load_cycle_table(Path("./jason_3_f.csv"))
    cycle_table_em_f  = CycleBasedRead.load_cycle_table(Path("./jason_3_em_f.csv"))

    # --- loop over cycles ---
    for cycle_number in cycle_list:
        print("\n" + "=" * 60)
        print(f"[INFO] Start processing cycle {cycle_number}")

        # choose correct cycle table
        df_cycle = cycle_table_f if cycle_number <= 227 else cycle_table_em_f
        cycle_start, cycle_end = CycleBasedRead.get_cycle_time_window(df_cycle, cycle_number)
        # --- Load full-cycle data ---
        df_alt_cycle = CycleBasedRead.load_altimetry_cycle(proc_sate, paths["alt_vtec"], cycle_start, cycle_end)
        df_doris_cycle = CycleBasedRead.load_doris_cycle(proc_sate, paths["doris_vtec"], cycle_start, cycle_end)
        df_station_cycle = CycleBasedRead.load_station_cycle(paths["station"], cycle_start, cycle_end)
        gim_cycle = CycleBasedRead.load_gim_cycle(paths["gim"], cycle_start, cycle_end)

        all_pass_results = []
        unique_passes = sorted(df_alt_cycle["pass_in_cycle"].unique())

        # --- process each pass of this cycle in parallel ---
        with ProcessPoolExecutor(max_workers=10) as executor:
            future_to_pid = {
                executor.submit(
                    process_single_pass,
                    pid,
                    df_alt_cycle,
                    df_doris_cycle,
                    df_station_cycle,
                    gim_cycle,
                    settings,
                    cycle_number,
                ): pid
                for pid in unique_passes
            }

            for future in tqdm(
                as_completed(future_to_pid),
                total=len(future_to_pid),
                desc=f"Cycle {cycle_number} (passes)",
                unit="pass",
            ):
                pid = future_to_pid[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[PASS {pid}] error: {e}")
                    continue

                if result is not None and not result.empty:
                    all_pass_results.append(result)

        # --- storage for this cycle ---
        if all_pass_results:
            df_all = pd.concat(all_pass_results, axis=0).reset_index(drop=True)

            # min_pass = int(df_all["pass_in_cycle"].min())
            # max_pass = int(df_all["pass_in_cycle"].max())

            save_dir = paths["dbav"] / f"{proc_sate} / cycle_unit / {settings['distance_station']}km"
            os.makedirs(save_dir, exist_ok=True)

            out_file = save_dir / f"cycle{cycle_number:03d}.pkl"
            df_all.to_pickle(out_file)
            print(f"[INFO] Saved cycle-based pass results to {out_file}")
        else:
            print(f"[INFO] No pass results for cycle {cycle_number}.")

    print(f"\n[INFO] Total elapsed time: {time.time() - start_time:.2f} seconds")

