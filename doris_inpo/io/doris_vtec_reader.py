from pathlib import Path
import pandas as pd

def load_doris_cycle(
    doris_vtec_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load DORIS-derived VTEC data covering a full satellite cycle.

    Parameters
    ----------
    satellite : str
        Satellite identifier.
    doris_vtec_dir : Path
        Root directory of DORIS VTEC products.
    cycle_start : pandas.Timestamp
        Start time of the cycle.
    cycle_end : pandas.Timestamp
        End time of the cycle.

    Returns
    -------
    pandas.DataFrame
        Concatenated DORIS observations within the cycle time window.
    """
    doris_list = []

    start_date = cycle_start.normalize()
    end_date = cycle_end.normalize()
    total_days = (end_date - start_date).days + 1

    for day_offset in range(total_days):
        date = start_date + pd.Timedelta(days=day_offset)
        year = date.year
        doy = date.timetuple().tm_yday

        doris_file = doris_vtec_dir/f"{year}/DOY{doy:03d}.pickle"
        if not doris_file.exists():
            print(f"[DORIS] Skip DOY{doy:03d} — file not found")
            continue

        df_d = pd.read_pickle(doris_file)
        df_d = df_d[
            (df_d["obs_epoch"] >= cycle_start) &
            (df_d["obs_epoch"] <= cycle_end)
        ]

        if not df_d.empty:
            doris_list.append(df_d)

    if not doris_list:
        raise RuntimeError("No DORIS data found for this cycle time window.")

    df_doris = pd.concat(doris_list, axis=0)
    return df_doris.dropna(subset=["vtec"]).reset_index(drop=True)