import re
from pathlib import Path
import pandas as pd

SP3_RANGE_PATTERN = re.compile(r"\.b(\d{5})\.e(\d{5})\.")


def tai_epochs_to_utc(epochs):
    """Convert TAI labels to UTC for SINEX/GIM labels and file selection.

    UTC leap seconds, which numpy cannot represent, belong to the final
    nanosecond of their UTC day. Physical interpolation must remain in TAI.
    """
    import numpy as np
    from astropy.time import Time

    index = pd.DatetimeIndex(pd.to_datetime(epochs, errors="coerce"))
    if index.tz is not None or index.isna().any():
        raise ValueError("TAI epochs must be finite timezone-naive labels.")
    tai = index.to_numpy(dtype="datetime64[ns]")
    utc = Time(tai, format="datetime64", scale="tai").utc
    utc.precision = 9
    try:
        return np.asarray(utc.datetime64, dtype="datetime64[ns]")
    except ValueError as exc:
        if "Seconds out of range" not in str(exc):
            raise
    labels = np.atleast_1d(utc.isot)
    leap = np.asarray([label[17:19] == "60" for label in labels])
    if not leap.any():
        raise ValueError("UTC conversion failed without a leap second.")
    result = np.empty(tai.shape, dtype="datetime64[ns]")
    result[~leap] = np.asarray(utc[~leap].datetime64, dtype="datetime64[ns]")
    result[leap] = [np.datetime64(label[:10], "D") + np.timedelta64(1, "D")
                    - np.timedelta64(1, "ns") for label in labels[leap]]
    return result


def correct_receiver_epochs(observations: pd.DataFrame) -> pd.DataFrame:
    """Apply the RINEX receiver offset once, preserving legacy column names.

    Old Stage1 tables retain the raw receiver epoch and its offset. New tables
    mark the corrected TAI identity so subsequent stages do not add it twice.
    Unknown-time tables without an offset are not relabeled as corrected TAI.
    """
    import numpy as np

    result = observations.copy()
    epochs = pd.to_datetime(result["obs_epoch"], errors="coerce")
    if epochs.isna().any() or epochs.dt.tz is not None:
        raise ValueError("Observation epochs must be finite timezone-naive labels.")
    if result.attrs.get("obs_epoch_receiver_clock_corrected") is not True:
        if "receiver_clock_offset" not in result:
            raise ValueError("Receiver clock offset is required for unverified epochs.")
        offset = pd.to_numeric(result["receiver_clock_offset"], errors="coerce")
        if not np.isfinite(offset.to_numpy(float)).all():
            raise ValueError("Receiver clock offsets must be finite seconds.")
        epochs = epochs + pd.to_timedelta(offset, unit="s")
    result["obs_epoch"] = epochs
    result.attrs.update(obs_epoch_time_scale="TAI", obs_epoch_receiver_clock_corrected=True)
    return result

def parse_sp3_time(yydoy: int) -> pd.Timestamp:
    """
    Parse SP3 YYDOY time code into pandas Timestamp.
    Example: 24001 -> 2024-01-01
    """
    year = 2000 + yydoy // 1000
    doy = yydoy % 1000
    return (
        pd.Timestamp(year=year, month=1, day=1)
        + pd.Timedelta(days=doy - 1)
    )


def parse_sinex_time(time_str: str) -> pd.Timestamp:
    """
    Parse SINEX time format: YY:DOY:SECOD
    """
    if not time_str or time_str.strip() in {"000:000:00000", "00:000:00000"}:
        return pd.NaT

    yy, doy, sec = time_str.split(":")
    year = int(yy)
    year += 2000 if year < 80 else 1900

    return (
        pd.Timestamp(year=year, month=1, day=1)
        + pd.Timedelta(days=int(doy) - 1)
        + pd.Timedelta(seconds=float(sec))
    )


def find_covering_sp3_files_from_dir(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    folder: Path,
) -> list[Path]:
    """
    Find SP3 files whose time coverage overlaps with [start_time, end_time].
    """
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory.")

    files: list[Path] = []

    for sp3_file in folder.glob("*.sp3.001"):
        if sp3_file.name.startswith("._"):
            continue

        m = SP3_RANGE_PATTERN.search(sp3_file.name)
        if not m:
            continue

        b = parse_sp3_time(int(m.group(1)))
        e = parse_sp3_time(int(m.group(2)))

        if e >= start_time and b <= end_time:
            files.append(sp3_file)

    return sorted(files, key=lambda p: p.name)


def find_covering_gim_file_from_dir(
    gim_dir: Path,
    year: int,
    doy: int,
) -> Path | None:
    """
    Find a GIM (IONEX) file covering the given year and DOY
    from a yearly GIM directory.
    """

    gim_dir_year = gim_dir / f"{year}"
    if not gim_dir_year.is_dir():
        return None

    file_list = [f.name for f in gim_dir_year.iterdir() if f.is_file()]

    # New IGS final GIM (INX-based)
    inx_file = next(
        (
            f for f in file_list
            if f.startswith(f"IGS0OPSFIN_{year}{doy:03d}")
            and f.endswith("GIM.INX")
        ),
        None,
    )

    # Classic igsg IONEX
    igsg_file = next(
        (
            f for f in file_list
            if f.startswith(f"igsg{doy:03d}0.{str(year % 100).zfill(2)}i")
        ),
        None,
    )

    gim_file = inx_file or igsg_file
    if gim_file is None:
        return None

    return gim_dir_year / gim_file
