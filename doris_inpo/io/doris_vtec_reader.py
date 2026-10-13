from pathlib import Path
from math import isfinite
import pandas as pd

def load_doris_cycle(
    doris_vtec_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
    *,
    require_corrected_tai: bool = False,
    time_margin_sec: float = 0.0,
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
    time_margin_sec : float
        Nonnegative reading halo on both sides, in seconds. The default
        preserves the historical cycle-only selection.

    Returns
    -------
    pandas.DataFrame
        Concatenated DORIS observations within the requested reading window.
    """
    doris_list = []
    time_identities = set()

    if not isfinite(time_margin_sec) or time_margin_sec < 0:
        raise ValueError("time_margin_sec must be finite and nonnegative")
    margin = pd.Timedelta(seconds=time_margin_sec)
    read_start = cycle_start - margin
    read_end = cycle_end + margin
    start_date = read_start.normalize()
    end_date = read_end.normalize()
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
            (df_d["obs_epoch"] >= read_start) &
            (df_d["obs_epoch"] <= read_end)
        ]

        if not df_d.empty:
            corrected = (df_d.attrs.get("obs_epoch_time_scale") == "TAI"
                         and df_d.attrs.get("obs_epoch_receiver_clock_corrected") is True)
            if require_corrected_tai and not corrected:
                raise ValueError(f"DORIS VTEC lacks corrected-TAI evidence; rebuild Stage1/2: {doris_file}")
            time_identities.add(corrected)
            doris_list.append(df_d)

    if not doris_list:
        raise RuntimeError("No DORIS data found for this cycle time window.")
    if len(time_identities) != 1:
        raise ValueError("Cannot mix historical unknown-time and corrected-TAI DORIS VTEC.")

    df_doris = pd.concat(doris_list, axis=0)
    result = df_doris.dropna(subset=["vtec"]).reset_index(drop=True)
    if True in time_identities:
        result.attrs.update(obs_epoch_time_scale="TAI", obs_epoch_receiver_clock_corrected=True)
    else:
        result.attrs["obs_epoch_time_scale"] = "legacy_unspecified"
    return result
