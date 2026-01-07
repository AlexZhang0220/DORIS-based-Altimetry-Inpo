from pathlib import Path
import pandas as pd
from doris_vtec.io.ionex_reader import read_ionex_file
from doris_vtec.io.utils import find_covering_gim_file_from_dir

def load_gim_cycle(
    gim_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
):
    """
    Load all IONEX (GIM) files required for a satellite cycle.

    Returns
    -------
    dict[(year, doy) -> dict]
        Each entry contains:
        - "gim": dict[datetime -> 2D TEC map]
        - "lat": latitude grid
        - "lon": longitude grid
    """
    gim_cycle = {}

    start_date = cycle_start.normalize()
    end_date = cycle_end.normalize()
    total_days = (end_date - start_date).days + 1

    for day_offset in range(total_days):
        date = start_date + pd.Timedelta(days=day_offset)
        year = date.year
        doy = date.timetuple().tm_yday

        ion_file = find_covering_gim_file_from_dir(gim_dir, year, doy)
        if not ion_file.exists():
            print(f"[GIM] Skip {year}-DOY{doy:03d} — file not found")
            continue

        gim, lat_grid, lon_grid = read_ionex_file(ion_file)

        for epoch, tec_map in gim.items():
            gim_cycle[epoch] = tec_map

        if not gim_cycle:
            raise RuntimeError("No GIM files loaded for this cycle.")

    return gim_cycle, lat_grid, lon_grid