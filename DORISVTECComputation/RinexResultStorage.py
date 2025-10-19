from ObsStorage import DORISStorage
from StationStorage import StationStorage
from pathlib import Path
from tqdm.auto import tqdm
import re
import time
import pickle
import pandas as pd
import georinex as gr
import xarray as xr

def yydoy_to_timestamp(yydoy: int) -> pd.Timestamp:
    year = 2000 + yydoy // 1000
    doy = yydoy % 1000
    return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)

def find_covering_sp3_files_from_dir(start_time: pd.Timestamp, end_time: pd.Timestamp, folder_path: str) -> list:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Provided path {folder_path} is not a valid directory.")
    
    matching_files = []
    for sp3_file in folder.glob("*.sp3.001"):
        match = re.search(r'\.b(\d{5})\.e(\d{5})\.', sp3_file.name)
        if match:
            b_yydoy = int(match.group(1))
            e_yydoy = int(match.group(2))
            file_start = yydoy_to_timestamp(b_yydoy)
            file_end = yydoy_to_timestamp(e_yydoy)
            if file_end >= start_time and file_start <= end_time:
                matching_files.append(folder_path + '/' + sp3_file.name)
    return sorted(matching_files) # sort by filedate

# def regenerate_daily_obs_with_margin(
#     start_time: pd.Timestamp,
#     end_time: pd.Timestamp,
#     data_dir: Path,
#     margin_minutes: int = 10,
# ):
#     '''
#     add more data towards the edge of the day, so continous passes can be formed there
#     '''    
#     year = start_time.year
#     margin = pd.Timedelta(minutes=margin_minutes)

#     def load_day_data(doy: int) -> pd.DataFrame:
#         path = data_dir / f"{year}/DOY{doy:03d}.pkl"
#         try:
#             with open(path, "rb") as f:
#                 return pickle.load(f)
#         except FileNotFoundError:
#             return None

#     for day in pd.date_range(start=start_time, end=end_time, freq="D"):
        
#         doy = day.dayofyear
#         prev_obs = load_day_data(doy - 1)
#         curr_obs = load_day_data(doy)
#         next_obs = load_day_data(doy + 1)

#         if curr_obs is None or curr_obs.storage.empty:
#             continue

#         all_data = pd.concat([
#             prev_obs.storage if prev_obs else pd.DataFrame(),
#             curr_obs.storage,
#             next_obs.storage if next_obs else pd.DataFrame()
#         ], ignore_index=True)

#         all_data = all_data.drop_duplicates().reset_index(drop=True)

#         df_window = all_data[
#             (all_data["obs_epoch"] >= day - margin) &
#             (all_data["obs_epoch"] < day + pd.Timedelta(days=1) + margin)
#         ].sort_values(by=["station_code", "obs_epoch"])

#         new_obs = DORISStorage()
#         new_obs.storage = df_window
#         new_obs.stations = curr_obs.stations

#         out_path = data_dir / f"{year}/DOY{doy:03d}.pkl"
#         with open(out_path, "wb") as f:
#             pickle.dump(new_obs, f)

satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']

def iter_blocks(start_dt: pd.Timestamp, end_dt: pd.Timestamp, max_block_days: int = 10):
    """
    Split a given time range into consecutive blocks, each up to `max_block_days` long.
    Automatically handles the final (possibly shorter) block.
    """
    cur = start_dt
    one_day = pd.Timedelta(days=1)
    while cur <= end_dt:
        block_end = min(cur + pd.Timedelta(days=max_block_days - 1), end_dt)
        yield cur, block_end
        cur = block_end + one_day


if __name__ == '__main__':
    start_time = time.time()

    # === Input parameters ===
    year, month, day = 2024, 5, 1
    proc_days = 80 
    proc_sate = satellite_list[1]

    start_dt = pd.Timestamp(year, month, day)
    end_dt = start_dt + pd.Timedelta(days=proc_days - 1)

    # === File paths ===
    sinex_path = f"/Volumes/T7Shield/DORISData/DORISInput/sinex/dpod2020_041.snx"
    sp3_root = f"/Volumes/T7Shield/DORISData/DORISInput/sp3/{proc_sate}"
    rinex_root = f"/Volumes/T7Shield/DORISData/DORISInput/rinexobs/{proc_sate}"
    obs_dir = Path(f"/Volumes/T7Shield/DORISData/DORISObsStorage/{proc_sate}")

    # === Progress bar ===
    pbar = tqdm(desc=f"Processing {proc_sate}", unit="day")

    # === Process data in 10-day blocks ===
    for block_start, block_end in iter_blocks(start_dt, end_dt):
        # 1) Load station coordinates (SINEX)
        stations = StationStorage()
        stations.read_sinex(sinex_path, block_start, block_end)
        if not stations.storage:
            raise ValueError(f"No station data found in SINEX for {block_start} to {block_end}")    
        
        # 2) Load satellite orbit data (SP3)
        matching_sp3 = find_covering_sp3_files_from_dir(block_start, block_end, sp3_root)
        sp3_file_list = [gr.load(file) for file in matching_sp3]
        if not sp3_file_list:
            raise ValueError(f"No SP3 files found covering {block_start} to {block_end}")
        orbit_data = xr.concat(sp3_file_list, dim="time")

        # 3) Process daily DORIS observation files within this block
        cur = block_start
        while cur <= block_end:
            y = cur.year
            doy = cur.dayofyear

            rinex_path = rinex_root / f"{y}/{proc_sate}rx{str(y)[-2:]}{doy:03d}.001"
            if not Path(rinex_path).is_file():
                print(f"RINEX file not found: {rinex_path}, skipping...")
                cur += pd.Timedelta(days=1)
                pbar.update(1)
                continue

            obs = DORISStorage()
            obs.read_rinex_300(rinex_path, orbit_data, stations)

            output_path = obs_dir / f"{y}/DOY{doy:03d}.pkl"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                pickle.dump(obs, f)

            cur += pd.Timedelta(days=1)
            pbar.update(1)

    pbar.close()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")