from pathlib import Path
import pandas as pd
from readFile import read_ionFile
import numpy as np
import tools
# ------------------------------------------------------------------------------
# 1. Cycle helpers
# ------------------------------------------------------------------------------

def load_cycle_table(cycle_csv_path: Path) -> pd.DataFrame:
    """
    Load Jason-3 cycle metadata table (like jason_3_f.csv) and build start/end datetimes.
    """
    df_cyc = pd.read_csv(cycle_csv_path)

    df_cyc["start_datetime"] = pd.to_datetime(
        df_cyc["start_day"] + " " + df_cyc["start_time"]
    )
    df_cyc["end_datetime"] = pd.to_datetime(
        df_cyc["end_date"] + " " + df_cyc["end_time"]
    )
    return df_cyc


def get_cycle_time_window(df_cycle: pd.DataFrame, cycle_number: int):
    """
    Return start/end datetime for a given cycle_number.
    """
    row = df_cycle.loc[df_cycle["cycle_number"] == cycle_number]
    if row.empty:
        raise ValueError(f"cycle_number={cycle_number} not found in cycle table.")

    row = row.iloc[0]
    start_dt = row["start_datetime"]
    end_dt = row["end_datetime"]
    return start_dt, end_dt


# ------------------------------------------------------------------------------
# 2. Loading data for a whole cycle
# ------------------------------------------------------------------------------

def load_altimetry_cycle(
    proc_sate: str,
    alt_vtec_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load altimetry data for a whole cycle (spanning several DOYs).
    Files are assumed to be stored as alt_vtec_dir/proc_sate/YYYY/DOYxxx.pkl
    """
    alt_list = []

    # pass is not a good name

    start_date = cycle_start.normalize()
    end_date = cycle_end.normalize()
    total_days = (end_date - start_date).days + 1

    for day_offset in range(total_days):
        date = start_date + pd.Timedelta(days=day_offset)
        year = date.year
        doy = date.timetuple().tm_yday

        alt_file = alt_vtec_dir / f"{proc_sate}/{year}/DOY{doy:03d}.pkl"
        if not alt_file.exists():
            print(f"[ALT] Skip DOY{doy:03d} — file not found")
            continue

        df_alt = pd.read_pickle(alt_file)

        # Keep only data inside the cycle time window
        df_alt = df_alt[
            (df_alt["obs_epoch"] >= cycle_start) &
            (df_alt["obs_epoch"] <= cycle_end)
        ]
        if df_alt.empty:
            continue

        alt_list.append(df_alt)

    if not alt_list:
        raise RuntimeError("No altimetry data found for this cycle time window.")

    df_alt_cycle = pd.concat(alt_list, axis=0).reset_index(drop=True).drop(columns=["VTEC_ns","ascend"])
    df_alt_cycle.rename(
        columns={"pass": "pass_in_cycle"},
        inplace=True
    )
    return df_alt_cycle


def load_doris_cycle(
    proc_sate: str,
    doris_vtec_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load DORIS data for a whole cycle, merging across DOYs.
    Files: doris_vtec_dir/proc_sate/YYYY/DOYxxx.pkl
    """
    doris_list = []

    start_date = cycle_start.normalize()
    end_date = cycle_end.normalize()
    total_days = (end_date - start_date).days + 1

    for day_offset in range(total_days):
        date = start_date + pd.Timedelta(days=day_offset)
        year = date.year
        doy = date.timetuple().tm_yday

        doris_file = doris_vtec_dir / f"{proc_sate}/{year}/DOY{doy:03d}.pkl"
        if not doris_file.exists():
            print(f"[DORIS] Skip DOY{doy:03d} — file not found")
            continue

        df_d = pd.read_pickle(doris_file)

        df_d = df_d[
            (df_d["obs_epoch"] >= cycle_start) &
            (df_d["obs_epoch"] <= cycle_end)
        ]
        if df_d.empty:
            continue

        doris_list.append(df_d)

    if not doris_list:
        raise RuntimeError("No DORIS data found for this cycle time window.")

    df_doris_cycle = pd.concat(doris_list, axis=0).reset_index(drop=True)
    return df_doris_cycle


def load_station_cycle(
    station_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load station coordinates for all DOYs covering a cycle and keep unique stations.
    Assumes files: station_dir/YYYY/DOYxxx.pkl
    Station coordinates are assumed to be time-invariant.
    """
    sta_list = []
    start_date = cycle_start.normalize()
    end_date = cycle_end.normalize()
    total_days = (end_date - start_date).days + 1

    for day_offset in range(total_days):
        date = start_date + pd.Timedelta(days=day_offset)
        year = date.year
        doy = date.timetuple().tm_yday

        sta_file = station_dir / f"{year}/DOY{doy:03d}.pkl"
        if not sta_file.exists():
            print(f"[STA] Skip DOY{doy:03d} — station file not found")
            continue

        df_sta = pd.read_pickle(sta_file)
        sta_list.append(df_sta)

    if not sta_list:
        raise RuntimeError("No station data found for this cycle.")

    df_station_cycle = pd.concat(sta_list, axis=0)

    # Keep one record per station_code (coordinates are constant over time)
    df_station_cycle = (
        df_station_cycle
        .drop_duplicates(subset=["station_code"], keep="last")
        .reset_index(drop=True)
    )
    return df_station_cycle


def load_gim_cycle(
    gim_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Load all GIM files needed for a given cycle.

    Returns
    -------
    gim_cycle : dict[(year, doy) -> gim_vtec]
        gim_vtec is the output of read_ionFile, e.g.
        tecuList[:, ::-1, :] * 10 ** exponent
    """
    gim_cycle: dict[tuple[int, int], np.ndarray] = {}

    start_date = cycle_start.normalize()
    end_date = cycle_end.normalize()
    total_days = (end_date - start_date).days + 1

    for day_offset in range(total_days):
        date = start_date + pd.Timedelta(days=day_offset)
        year = date.year
        doy = date.timetuple().tm_yday

        key = (year, doy)
        ion_file = tools.get_gim_filename(gim_dir, year, doy)
        if not ion_file.exists():
            print(f"[GIM] Skip {year}-DOY{doy:03d} — file not found")
            continue

        gim_vtec = read_ionFile(str(ion_file))
        gim_cycle[key] = gim_vtec

    if not gim_cycle:
        raise RuntimeError("No GIM files loaded for this cycle.")

    return gim_cycle