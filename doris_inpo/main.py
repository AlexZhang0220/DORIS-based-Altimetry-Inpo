
import os
import time
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append(".")
sys.path.append("..")
from doris_inpo.io.cycleinfo_reader import load_cycle_table, get_cycle_time_window
from doris_inpo.io.altimetry_vtec_reader import load_altimetry_cycle
from doris_inpo.io.doris_vtec_reader import load_doris_cycle
from doris_inpo.io.gim_vtec_reader import load_gim_cycle
from doris_vtec.stage2_stec2vtec.leveling.gim_interpolation import get_gim_vtec
from doris_vtec.io.sinex_reader import read_sinex_file

from doris_inpo.interpolation.prefilter import altimetry_distance_mask
from doris_inpo.interpolation.doris_based_alt_inpo import altimetry_inpo

from doris_inpo.config import DEFAULT_INPO_CONFIG

SECONDS_PER_YEAR = 365.25 * 86400.0


def process_single_pass(
    pid,
    df_alt_cycle,
    df_doris_cycle,
    settings,
    cycle_number,
):
    """
    Process a single altimetry pass.
    """

    df_alt_pass = df_alt_cycle[
        df_alt_cycle["pass_in_cycle"] == pid
    ].copy()
    if df_alt_pass.empty:
        return None

    # time window for DORIS
    t0 = df_alt_pass["obs_epoch"].min()
    t1 = df_alt_pass["obs_epoch"].max()

    df_doris_pass = df_doris_cycle[
        (df_doris_cycle["obs_epoch"] >= t0 - pd.Timedelta(minutes=5)) &
        (df_doris_cycle["obs_epoch"] <= t1 + pd.Timedelta(minutes=5))
    ]
    if df_doris_pass.empty:
        return None

    # interpolation
    df_interp = altimetry_inpo(
        df_alt_pass,
        df_doris_pass,
        settings
    )
    if df_interp.empty:
        return None

    # bookkeeping
    df_interp["cycle_number"] = cycle_number
    df_interp["pass_in_cycle"] = pid

    return df_interp



# =============================================================================
# Main pipeline
# =============================================================================
if __name__ == "__main__":

    t_all = time.time()

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    satellite = "ja3"
    cycle_list = range(70, 71)

    # Input
    sinex_path = Path("/Volumes/T7Shield/DORISData/DORISInput/sinex/dpod2020_current.snx")
    gim_root = Path("/Volumes/T7Shield/DORISData/DORISInput/IGSGIM")
    doris_vtec_root = Path(f"/Volumes/T7Shield/DORISData/DORISVTECStorage/{satellite}")
    alt_vtec_root = Path(f"/Volumes/T7Shield/AltimetryData/AltimetryVTECStorage/{satellite}")


    # Output
    dbav_root = Path(f"/Volumes/T7Shield/DORISData/DBAVStorage/{satellite}")

    max_workers = min(10, os.cpu_count() - 1)

    # :(
    cycle_table_f = load_cycle_table(Path("./jason_3_f.csv"))
    cycle_table_em_f = load_cycle_table(Path("./jason_3_em_f.csv"))

    df_sta = read_sinex_file(sinex_path)

    # =============================================================================
    # Loop over cycles
    # =============================================================================
    for cycle_number in cycle_list:

        print("\n" + "=" * 60)
        print(f"[PIPELINE] Cycle {cycle_number}")

        # --- select correct cycle table ---
        df_cycle = cycle_table_f if cycle_number <= 227 else cycle_table_em_f
        cycle_start, cycle_end = get_cycle_time_window(df_cycle, cycle_number)

        # ---------------------------------------------------------------------
        # Load cycle-level data
        # ---------------------------------------------------------------------
        df_alt_cycle = load_altimetry_cycle(
            alt_vtec_dir=alt_vtec_root, 
            cycle_start=cycle_start, 
            cycle_end=cycle_end
        )
        df_doris_cycle = load_doris_cycle(
            doris_vtec_dir=doris_vtec_root, 
            cycle_start=cycle_start, 
            cycle_end=cycle_end
        )
        gim_cycle, lat_grid, lon_grid = load_gim_cycle(
            gim_dir=gim_root, 
            cycle_start=cycle_start, 
            cycle_end=cycle_end
        )

        # ---------------------------------------------------------------------
        # Compute station coordinates and gim reference at cycle start (once per cycle)
        # ---------------------------------------------------------------------
        sta_records = []
        t_ref = cycle_start

        for station_code in df_doris_cycle["station_code"].unique():

            sta_group = df_sta[df_sta["station_code"] == station_code]

            valid = (
                (sta_group["start_obs"] <= t_ref) &
                (t_ref <= sta_group["end_obs"])
            )

            if valid.sum() != 1:
                continue

            sol = sta_group.loc[valid].iloc[0]

            dt_year = (
                (t_ref - sol["ref_epoch"]).total_seconds()
                / SECONDS_PER_YEAR
            )

            sta_records.append(
                {
                    "station_code": station_code,
                    "sta_x": sol["x"] + sol["vx"] * dt_year,
                    "sta_y": sol["y"] + sol["vy"] * dt_year,
                    "sta_z": sol["z"] + sol["vz"] * dt_year,
                }
            )

        df_station_cycle = pd.DataFrame(sta_records)
        sta_xyz = df_station_cycle[["sta_x", "sta_y", "sta_z"]].to_numpy()

        df_alt_cycle = altimetry_distance_mask(
            df_alt_cycle, 
            sta_xyz, 
            max_km=DEFAULT_INPO_CONFIG["max_station_distance"])

        df_alt_cycle['gim_vtec'] = get_gim_vtec(gim_cycle, lat_grid, lon_grid, df_alt_cycle)
        # ---------------------------------------------------------------------
        # Pass-level multiprocessing
        # ---------------------------------------------------------------------
        results = []

        unique_passes = sorted(df_alt_cycle["pass_in_cycle"].unique())

        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            future_map = {
                executor.submit(
                    process_single_pass,
                    pid,
                    df_alt_cycle,
                    df_doris_cycle,
                    DEFAULT_INPO_CONFIG,
                    cycle_number,
                ): pid
                for pid in unique_passes
            }

            for future in tqdm(
                as_completed(future_map),
                total=len(future_map),
                desc=f"Cycle {cycle_number}",
                unit="pass",
            ):
                try:
                    res = future.result()
                except Exception as e:
                    pid = future_map[future]
                    print(f"[Cycle {cycle_number} | Pass {pid}] ERROR: {e}")
                    continue

                if res is not None and not res.empty:
                    results.append(res)

        # ---------------------------------------------------------------------
        # Save cycle result
        # ---------------------------------------------------------------------
        if results:
            df_out = pd.concat(results, axis=0).reset_index(drop=True)
            df_out = df_out.sort_values(by=["obs_epoch"]).reset_index(drop=True)

            out_dir = paths["dbav"]/"cycle_unit"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir/f"cycle{cycle_number:03d}.pickle"
            df_out.to_pickle(out_file)

        else:
            print(f"[PIPELINE] No valid passes for cycle {cycle_number}")

    print(f"\n[PIPELINE] Total elapsed time: {time.time() - t_all:.1f} s")