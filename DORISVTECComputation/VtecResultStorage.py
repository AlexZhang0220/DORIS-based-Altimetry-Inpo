from CycleSlipDetection import detect_passes
from tools import get_igs_vtec, get_gim_filename
from readFile import read_ionFile
from pathlib import Path
from tqdm.auto import tqdm
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# === Configuration ===
satellite_list = ['ja2', 'ja3', 's3a', 's3b', 'srl']
range_ratio_list = [0.925]

# def plot_demo_level(pass_df, max_row, corrected_stec):
#     plt.rcParams.update({
#         "font.family": "Arial",
#         "font.size": 12,
#         "axes.labelsize": 14,
#         "axes.titlesize": 14,
#         "xtick.labelsize":12,
#         "ytick.labelsize":12,
#         "legend.fontsize":12,
#         "axes.linewidth": 1.0,
#     })

#     # ---- relative time (seconds) ----
#     t0 = pass_df["obs_epoch"].iloc[0]
#     time_sec = (pass_df["obs_epoch"] - t0).dt.total_seconds()

#     # ---- unleveled STEC ----
#     unleveled_stec = pass_df["STEC"] - pass_df["STEC"].iloc[0]

#     # ---- max elevation epoch ----
#     idx_max = pass_df["elevation"].idxmax()
#     t_max   = time_sec.loc[idx_max]
#     stec_max = unleveled_stec.loc[idx_max]
#     gim_stec = max_row.igs_vtec

#     # =========================
#     # Figure (single column)
#     # =========================
#     fig, ax1 = plt.subplots(figsize=(6.6, 5.2))
#     ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)

#     # ---- STEC curves ----
#     ax1.plot(
#         time_sec,
#         unleveled_stec,
#         color="#5fa8d3",
#         lw=2.8,
#         label="Unleveled DORIS STEC"
#     )

#     ax1.plot(
#         time_sec,
#         corrected_stec,
#         color="#1F77B4",
#         lw=2.8,
#         label="Leveled DORIS STEC"
#     )

#     # ---- reference point (max elevation) ----
#     ax1.scatter(
#         t_max,
#         stec_max,
#         s=40,
#         color="#FF7F0E",
#         zorder=4
#     )

#     ax1.scatter(
#         t_max,
#         gim_stec,
#         s=40,
#         color="#FF7F0E",
#         zorder=4,
#         label="GIM STEC (reference)"
#     )

#     # ---- arrow: leveling shift ----
#     ax1.annotate(
#         "",
#         xy=(t_max, gim_stec),
#         xytext=(t_max, stec_max),
#         arrowprops=dict(
#             arrowstyle="->",
#             lw=1.8,
#             color="black"
#         )
#     )

#     ax1.text(
#         t_max + 0.02 * (time_sec.max() - time_sec.min()),
#         0.5 * (gim_stec + stec_max),
#         "Leveling",
#         va="center"
#     )

#     ax1.set_xlabel("Time (s)")
#     ax1.set_ylabel("STEC (TECU)")

#     # ---- elevation (secondary axis, weak) ----
#     ax2 = ax1.twinx()
#     ax2.plot(
#         time_sec,
#         pass_df["elevation"],
#         color="red",
#         lw=2.8,
#         alpha=0.7,
#         label="Elevation"
#     )
#     ax2.set_ylim(0, 90)
#     ax2.set_ylabel("Elevation (deg)")

#     # ---- combined legend ----
#     h1, l1 = ax1.get_legend_handles_labels()
#     h2, l2 = ax2.get_legend_handles_labels()
#     handles1, labels1 = ax1.get_legend_handles_labels()
#     handles2, labels2 = ax2.get_legend_handles_labels()
#     ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

#     plt.tight_layout()
#     plt.savefig("passdemolevel.png", dpi=600)
#     plt.show()
#     plt.close()

def correct_pass_vtec(pass_df, max_row, igs_vtec, range_ratio):
    """
    Apply GIM-based VTEC leveling correction to a single pass.
    """
    igs_stec = igs_vtec * max_row['map_value']
    delta = igs_stec * range_ratio - max_row['STEC']
    print(igs_stec, delta)
    corrected_stec = pass_df['STEC'] + delta
    corrected_vtec = corrected_stec / pass_df['map_value']
    # plot_demo_level(pass_df, max_row, corrected_stec)
    pass_df = pass_df.copy()
    if pass_df.station_code.values[0] == 'AMVB':
        print('debug')
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
    start_date = pd.Timestamp(2018, 1, 1)
    proc_days = 10
    proc_sate = satellite_list[1]
    min_obs_count = 30    
    elev_thres = 10        
    range_ratio = range_ratio_list[0]

    columns_to_keep = [
        'obs_epoch', 'ipp_lat', 'ipp_lon',
        'STEC', 'elevation', 'map_value',
        'station_code', 'PRN'
    ]

    # === Main processing loop ===
    pbar = tqdm(range(proc_days), desc="VTEC Leveling", unit="day")
    for day_offset in pbar:
        process_epoch = start_date + pd.Timedelta(days=day_offset)
        year, doy = process_epoch.year, process_epoch.dayofyear

        # --- Load observation file ---
        obs_path = obs_dir / f"{proc_sate}/{year}/DOY{doy:03d}.parquet"
        if not obs_path.is_file():
            tqdm.write(f"Obs file not found: {obs_path}, skipping...")
            continue

        obs = pd.read_parquet(obs_path)

        df_station_unique = (
            obs[['station_code', 'sta_x', 'sta_y', 'sta_z']]
            .drop_duplicates(subset='station_code')
            .reset_index(drop=True)
        )

        save_dir = Path(f"/Volumes/T7Shield/DORISData/StationCoord/{year}")
        os.makedirs(save_dir, exist_ok=True)
        df_station_unique.to_pickle(save_dir / f"DOY{doy:03d}.pkl")
        
        # --- Detect and separate passes for all stations ---
        passes_all_stations = []
        for station_code, grouped_obs in obs.groupby('station_code'):
            passes_all_stations.append(
                detect_passes(grouped_obs, min_obs_count, elev_thres, columns_to_keep)
            )
        passes_all_stations = pd.concat(passes_all_stations, ignore_index=True)

        if passes_all_stations.empty:
            tqdm.write(f"No valid passes found for DOY{doy:03d}, skipping...")
            continue

        # --- Load GIM VTEC file ---
        ion_file = get_gim_filename(gim_dir, year, doy)
        if not ion_file:
            tqdm.write(f"GIM file not found for {year} DOY{doy:03d}, skipping...")
            continue

        # --- Read GIM data and compute IGS VTEC ---
        gim_vtec = read_ionFile(str(ion_file))

        # --- Get maximum elevation point for each pass ---
        max_elev_idx = passes_all_stations.groupby(['station_code', 'pass_id'])['elevation'].idxmax()
        df_max_elev = passes_all_stations.loc[max_elev_idx].reset_index(drop=True)
        df_max_elev['igs_vtec'] = get_igs_vtec(gim_vtec, df_max_elev)

        # Map each (station, pass_id) to its maximum elevation row
        max_elev_row_dict = {(row['station_code'], row['pass_id']): row for _, row in df_max_elev.iterrows()}

        # --- Apply VTEC leveling correction ---
        cor_p_all = []
        for (station_code, pass_id), pass_df in passes_all_stations.groupby(['station_code', 'pass_id']):
            max_row = max_elev_row_dict[(station_code, pass_id)]
            # if max_row.elevation < 60: continue
            igs_vtec = max_row['igs_vtec']
            cor_p = correct_pass_vtec(pass_df, max_row, igs_vtec, range_ratio)
            cor_p_all.append(cor_p)

        cor_pd_all = pd.concat(cor_p_all, ignore_index=True)

        # --- Save results ---
        vtec_path = vtec_dir / f"{proc_sate}/{year}/DOY{doy:03d}.pkl"

        vtec_path.parent.mkdir(parents=True, exist_ok=True)
        cor_pd_all.to_pickle(vtec_path)

    pbar.close()
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")