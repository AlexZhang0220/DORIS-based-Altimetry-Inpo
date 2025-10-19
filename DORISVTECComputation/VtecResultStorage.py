from CycleSlipDetection import detect_passes
from tools import get_igs_vtec
from readFile import read_ionFile
from pathlib import Path
from tqdm.auto import tqdm
import time
import pandas as pd
import pickle

# === Configuration ===
satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]


def correct_pass_vtec(pass_df, max_row, igs_vtec, range_ratio=1.0):
    """
    Apply GIM-based VTEC leveling correction to a single pass.
    """
    igs_stec = igs_vtec * max_row['map_value']
    delta = igs_stec * range_ratio - max_row['STEC']

    corrected_stec = pass_df['STEC'] + delta
    corrected_vtec = corrected_stec / pass_df['map_value']

    pass_df = pass_df.copy()
    pass_df['STEC'] = corrected_stec
    pass_df['VTEC'] = corrected_vtec
    return pass_df

if __name__ == '__main__':
    start_time = time.time()

    # === Directory setup ===
    obs_dir = Path("/Volumes/T7Shield/DORISData/DORISObsStorage")
    gim_dir = Path("/Volumes/T7Shield/DORISData/DORISInput/IGSGIM")
    vtec_dir = Path("/Volumes/T7Shield/DORISData/DORISVTECStorage")

    # === Processing parameters ===
    start_day = pd.Timestamp(2024, 5, 8)
    proc_days = 1
    proc_sate = satellite_list[1]
    min_obs_count = 30     # Minimum number of observations per pass
    elev_thres = 10        # Elevation angle threshold in degrees
    range_ratio = range_ratio_list[0]

    columns_to_keep = [
        'obs_epoch', 'ipp_lat', 'ipp_lon',
        'STEC', 'elevation', 'map_value',
        'station_code', 'PRN'
    ]

    # === Main processing loop ===
    pbar = tqdm(range(proc_days), desc="VTEC Leveling", unit="day")

    for day_offset in pbar:
        process_epoch = start_day + pd.Timedelta(days=day_offset)
        year, doy = process_epoch.year, process_epoch.dayofyear

        # --- Load observation file ---
        obs_path = obs_dir / f"{proc_sate}/{year}/DOY{doy:03d}.pkl"
        if not obs_path.is_file():
            tqdm.write(f"Obs file not found: {obs_path}, skipping...")
            continue

        with open(obs_path, 'rb') as f:
            obs = pickle.load(f)

        # --- Detect and separate passes for all stations ---
        passes_all_stations = []
        for station_code, grouped_obs in obs.storage.groupby('station_code'):
            passes_all_stations.append(
                detect_passes(grouped_obs, min_obs_count, elev_thres, columns_to_keep)
            )
        passes_all_stations = pd.concat(passes_all_stations, ignore_index=True)

        if passes_all_stations.empty:
            tqdm.write(f"No valid passes found for DOY{doy:03d}, skipping...")
            continue

        # --- Load GIM VTEC file ---
        gim_dir_year = gim_dir / f"{year}"
        file_list = [f.name for f in gim_dir_year.iterdir() if f.is_file()]
        inx_file = next((f for f in file_list if f.startswith(f"IGS0OPSFIN_{year}{doy:03d}") and f.endswith("GIM.INX")), None)
        igsg_file = next((f for f in file_list if f.startswith(f"igsg{doy:03d}0.{str(year % 100).zfill(2)}i")), None)
        ion_file = inx_file or igsg_file

        if not ion_file:
            tqdm.write(f"GIM file not found for {year} DOY{doy:03d}, skipping...")
            continue

        # --- Read GIM data and compute IGS VTEC ---
        gim_vtec = read_ionFile(ion_file) * 0.1
        gim_vtec = gim_vtec[:, ::-1, :]  # Flip longitude direction

        # --- Get maximum elevation point for each pass ---
        max_elev_idx = passes_all_stations.groupby(['station_code', 'pass_id'])['elevation'].idxmax()
        df_max_elev = passes_all_stations.loc[max_elev_idx].reset_index(drop=True)
        df_max_elev['igs_vtec'] = get_igs_vtec(gim_vtec, df_max_elev)

        # Map each (station, pass_id) to its maximum elevation row
        max_elev_row_dict = {(row['station_code'], row['pass_id']): row for _, row in df_max_elev.iterrows()}

        # --- Apply VTEC leveling correction ---
        corrected_passes_all = []
        for (station_code, pass_id), pass_df in passes_all_stations.groupby(['station_code', 'pass_id']):
            max_row = max_elev_row_dict[(station_code, pass_id)]
            igs_vtec = max_row['igs_vtec']
            corrected_pass = correct_pass_vtec(pass_df, max_row, igs_vtec, range_ratio)
            corrected_passes_all.append(corrected_pass)

        passes_all_stations_processed = pd.concat(corrected_passes_all, ignore_index=True)

        # --- Save results ---
        vtec_path = vtec_dir / f"{proc_sate}/{year}/DOY{doy:03d}.pkl"
        vtec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vtec_path, 'wb') as f:
            pickle.dump(passes_all_stations_processed, f)

    pbar.close()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")