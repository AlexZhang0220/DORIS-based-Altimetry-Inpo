from ObsStorage import DORISStorage
from StationStorage import StationStorage
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import re
import time
import pandas as pd
import georinex as gr
import xarray as xr
from gnsspy import read_sp3File

def yydoy_to_timestamp(yydoy: int) -> pd.Timestamp:
    year = 2000 + yydoy // 1000
    doy = yydoy % 1000
    return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=doy - 1)

def find_covering_sp3_files_from_dir(start_time: pd.Timestamp, end_time: pd.Timestamp, folder: Path) -> list:

    if not folder.is_dir():
        raise ValueError(f"Provided path {folder} is not a valid directory.")
    
    matching_files = []
    for sp3_file in folder.glob("*.sp3.001"):
        if not sp3_file.name.startswith("._"):
            match = re.search(r'\.b(\d{5})\.e(\d{5})\.', sp3_file.name)
            if match:
                b_yydoy = int(match.group(1))
                e_yydoy = int(match.group(2))
                file_start = yydoy_to_timestamp(b_yydoy)
                file_end = yydoy_to_timestamp(e_yydoy)
                if file_end >= start_time and file_start <= end_time:
                    matching_files.append(folder / sp3_file.name)
    return sorted(matching_files) # sort by filedate

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
    start_date = pd.Timestamp(2018, 1, 1)
    proc_days = 1
    proc_sate = satellite_list[1]
    end_date = start_date + pd.Timedelta(days=proc_days - 1)

    # === File paths ===
    sinex_path = Path(f"/Volumes/T7Shield/DORISData/DORISInput/sinex/dpod2020_current.snx")
    sp3_root = Path(f"/Volumes/T7Shield/DORISData/DORISInput/sp3/{proc_sate}")
    rinex_root = Path(f"/Volumes/T7Shield/DORISData/DORISInput/rinexobs/{proc_sate}")
    obs_dir = Path(f"/Volumes/T7Shield/DORISData/DORISObsStorage/{proc_sate}")

    # === Progress bar ===
    pbar = tqdm(desc=f"Processing {proc_sate}", unit="day")

    # === Process data in 10-day blocks ===
    for block_start, block_end in iter_blocks(start_date, end_date):
        # 1) Load station coordinates (SINEX)
        stations = StationStorage()
        stations.read_sinex(sinex_path, block_start, block_end)
        if not stations.storage:
            print(f"SINEX files not found covering: {block_start} to {block_end}, skipping...")
            pbar.update(10)
            continue        
        
        # 2) Load satellite orbit data (SP3)
        matching_sp3 = find_covering_sp3_files_from_dir(block_start, block_end, sp3_root)
        sp3_file_list = [gr.load(file) for file in matching_sp3]
        if not sp3_file_list:
            print(f"SP3 files not found covering: {block_start} to {block_end}, skipping...")
            pbar.update(10)
            continue
        orbit_data = xr.concat(sp3_file_list, dim="time")
        orbit_start_doy = (orbit_data.time.values[0] + np.timedelta64(1, 'D')).astype('datetime64[D]')
        cur = block_start if block_start > orbit_start_doy else pd.Timestamp(orbit_start_doy)
        
        # 3) Process daily DORIS observation files within this block
        while cur <= block_end:
            y = cur.year
            doy = cur.dayofyear

            rinex_path = rinex_root/f"{y}/{proc_sate}rx{str(y)[-2:]}{doy:03d}.001"
            if not Path(rinex_path).is_file():
                print(f"RINEX file not found: {rinex_path}, skipping...")
                cur += pd.Timedelta(days=1)
                pbar.update(1)
                continue

            obs = DORISStorage()
            obs.read_rinex_300(rinex_path, orbit_data, stations)

            output_path = obs_dir/f"{y}/DOY{doy:03d}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            obs.storage.to_parquet(output_path, compression="gzip")

            cur += pd.Timedelta(days=1)
            pbar.update(1)

    pbar.close()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")