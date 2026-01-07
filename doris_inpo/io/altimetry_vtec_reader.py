from pathlib import Path
import pandas as pd

def load_altimetry_cycle(
    alt_vtec_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load altimetry VTEC data covering a full satellite cycle.

    Altimetry data are assumed to be stored as daily pickle files,
    organized by satellite, year, and day-of-year (DOY).

    Parameters
    ----------
    satellite : str
        Satellite identifier (e.g. 'ja3', 's3a').
    alt_vtec_dir : Path
        Root directory of altimetry VTEC products.
    cycle_start : pandas.Timestamp
        Start time of the cycle.
    cycle_end : pandas.Timestamp
        End time of the cycle.

    Returns
    -------
    pandas.DataFrame
        Concatenated altimetry observations within the cycle time window.
    """
    alt_list = []

    start_date = cycle_start.normalize()
    end_date = cycle_end.normalize()
    total_days = (end_date - start_date).days + 1

    for day_offset in range(total_days):
        date = start_date + pd.Timedelta(days=day_offset)
        year = date.year
        doy = date.timetuple().tm_yday

        alt_file = Path(alt_vtec_dir/f"{year}/DOY{doy:03d}.pkl")
        if not alt_file.exists():
            print(f"[ALT] Skip DOY{doy:03d} — file not found")
            continue

        df_alt = pd.read_pickle(alt_file)
        df_alt = df_alt[
            (df_alt["obs_epoch"] >= cycle_start) &
            (df_alt["obs_epoch"] <= cycle_end)
        ]

        if not df_alt.empty:
            alt_list.append(df_alt)

    if not alt_list:
        raise RuntimeError("No altimetry data found for this cycle time window.")

    df_alt_cycle = (
        pd.concat(alt_list, axis=0)
        .reset_index(drop=True)
        .drop(columns=["VTEC_ns", "ascend"], errors="ignore")
    )

    df_alt_cycle = df_alt_cycle.dropna(subset=["VTEC"])
    df_alt_cycle.rename(columns={"pass": "pass_in_cycle"}, inplace=True)
    return df_alt_cycle