import re
from pathlib import Path
import pandas as pd

SP3_RANGE_PATTERN = re.compile(r"\.b(\d{5})\.e(\d{5})\.")

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
