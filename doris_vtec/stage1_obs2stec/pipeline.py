from pathlib import Path
import pandas as pd

# IO
from ..io.sp3_reader import read_sp3_files
from ..io.sinex_reader import read_sinex_file
from ..io.rinex_reader import read_rinex_file
from ..io.utils import find_covering_sp3_files_from_dir, correct_receiver_epochs

# Geometry
from .geometry.satellite import add_satellite_position
from .geometry.station import add_station_position
from .geometry.elevation import add_elevation
from .geometry.mapping import add_mapping_function
from .geometry.ipp import add_ipp_position

# Observation / physics
from .ionosphere.dion import add_doris_stec


def iter_blocks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    block_days: int,
):
    """
    Iterate over [start, end] in blocks of length block_days.
    """
    cur = start.normalize()
    end = end.normalize()

    while cur <= end:
        block_start = cur
        block_end = min(cur + pd.Timedelta(days=block_days - 1), end)
        yield block_start, block_end
        cur = block_end + pd.Timedelta(days=1)


def iter_days(start: pd.Timestamp, end: pd.Timestamp):
    """
    Iterate over days within one block.
    """
    cur = start.normalize()
    end = end.normalize()

    while cur <= end:
        yield cur
        cur += pd.Timedelta(days=1)


def load_block_aux(
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    *,
    sinex_file: Path,
    sp3_root: Path,
):
    """
    Load block-level auxiliary data (SP3 + SINEX).
    """
    df_sta = read_sinex_file(sinex_file)
    if df_sta.empty:
        raise RuntimeError("SINEX station DataFrame is empty.")

    sp3_files = find_covering_sp3_files_from_dir(
        block_start, block_end, sp3_root
    )
    if not sp3_files:
        raise RuntimeError(
            f"No SP3 files found covering {block_start} - {block_end}"
        )

    df_sp3 = read_sp3_files(sp3_files)

    return df_sp3, df_sta


def process_day(
    cur: pd.Timestamp,
    *,
    rinex_root: Path,
    satellite: str,
):
    """
    Read one-day RINEX observation.
    """
    year = cur.year
    doy = cur.dayofyear

    rinex_path = (rinex_root/f"{year}"/f"{satellite}rx{str(year)[-2:]}{doy:03d}.001")

    if not rinex_path.is_file():
        return None

    df_obs = read_rinex_file(rinex_path)
    if df_obs.empty:
        return None

    return df_obs


# ------------------------------------------------------------
# main block processor
# ------------------------------------------------------------
def process_block(
    block_start: pd.Timestamp,
    block_end: pd.Timestamp,
    *,
    sinex_file: Path,
    sp3_root: Path,
    rinex_root: Path,
    output_root: Path,
    satellite: str,
):
    """
    Process one block of DORIS observations.
    """
    # ----------------------------------------------------
    # block-level auxiliary data
    # ----------------------------------------------------
    df_sp3, df_sta = load_block_aux(
        block_start,
        block_end,
        sinex_file=sinex_file,
        sp3_root=sp3_root,
    )

    # ----------------------------------------------------
    # day-level processing
    # ----------------------------------------------------
    for cur in iter_days(block_start, block_end):

        # ---- read RINEX observations (obs-level) ----
        df_obs = process_day(
            cur,
            rinex_root=rinex_root,
            satellite=satellite,
        )

        if df_obs is None:
            continue

        # ------------------------------------------------
        # RINEX processing pipeline
        # ------------------------------------------------
        df_obs = (
            df_obs
            .pipe(correct_receiver_epochs)
            .pipe(add_satellite_position, df_sp3)
            .pipe(add_station_position, df_sta)
            .pipe(add_elevation)
            .pipe(add_mapping_function)
            .pipe(add_ipp_position)
            .pipe(add_doris_stec)
        )

        # ------------------------------------------------
        # output
        # ------------------------------------------------
        out_dir = output_root/f"{cur.year}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir/f"DOY{cur.dayofyear:03d}.pkl"
        df_obs.to_pickle(out_path)
