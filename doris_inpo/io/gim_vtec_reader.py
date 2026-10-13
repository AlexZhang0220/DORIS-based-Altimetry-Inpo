from pathlib import Path

import numpy as np
import pandas as pd

from doris_vtec.io.ionex_reader import GimMaps, read_ionex_file
from doris_vtec.io.utils import find_covering_gim_file_from_dir, tai_epochs_to_utc


def load_gim_cycle(
    gim_dir: Path,
    cycle_start: pd.Timestamp,
    cycle_end: pd.Timestamp,
    *,
    observation_time_system: str = "TAI",
):
    """Merge UTC-day IONEX files required by the requested physical interval.

    Conflicting shared-midnight maps use the file for that UTC day. Other
    overlapping conflicts and inconsistent grids are rejected.
    """
    if observation_time_system not in {"TAI", "UTC"}:
        raise ValueError("observation_time_system must be 'TAI' or 'UTC'")
    bounds = pd.DatetimeIndex([cycle_start, cycle_end])
    if observation_time_system == "TAI":
        bounds = pd.DatetimeIndex(tai_epochs_to_utc(bounds))
    start, end = bounds
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError("GIM cycle bounds must be finite and ordered")

    gim_cycle = GimMaps()
    map_days = {}
    loaded_days = set()
    lat_grid = lon_grid = None

    def load_day(day):
        nonlocal lat_grid, lon_grid
        day = pd.Timestamp(day).normalize()
        if day in loaded_days:
            return
        loaded_days.add(day)
        path = find_covering_gim_file_from_dir(Path(gim_dir), day.year, day.dayofyear)
        if path is None or not path.is_file():
            return
        gim, lat, lon = read_ionex_file(path)
        interval_seconds = getattr(gim, "interval_seconds", None)
        if interval_seconds is None:
            raise ValueError(f"Missing IONEX INTERVAL metadata: {path}")
        if not np.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError(f"Invalid IONEX INTERVAL metadata: {path}")
        if gim_cycle.interval_seconds is None:
            gim_cycle.interval_seconds = interval_seconds
        elif gim_cycle.interval_seconds != interval_seconds:
            raise ValueError(f"Inconsistent IONEX INTERVAL across files: {path}")
        if lat_grid is None:
            lat_grid, lon_grid = lat, lon
        elif not np.array_equal(lat, lat_grid) or not np.array_equal(lon, lon_grid):
            raise ValueError(f"Inconsistent IONEX grids across files: {path}")
        for epoch, tec_map in gim.items():
            epoch = pd.Timestamp(epoch)
            if np.shape(tec_map) != (len(lat_grid), len(lon_grid)):
                raise ValueError(f"IONEX map shape does not match grid: {path}")
            if epoch not in gim_cycle:
                gim_cycle[epoch] = tec_map
                map_days[epoch] = day
                continue
            previous_day = map_days[epoch]
            identical = np.array_equal(gim_cycle[epoch], tec_map, equal_nan=True)
            midnight = epoch == epoch.normalize()
            owner_day = epoch.normalize()
            if not identical and not midnight:
                raise ValueError(f"Conflicting IONEX maps at {epoch}: {previous_day.date()} and {day.date()}")
            if not identical and owner_day not in {previous_day, day}:
                raise ValueError(f"Conflicting midnight maps without the UTC-day owner: {epoch}")
            if day == owner_day:
                gim_cycle[epoch] = tec_map
                map_days[epoch] = day

    for day in pd.date_range(start.normalize(), end.normalize(), freq="D"):
        load_day(day)
    # Daily files need not include both temporal endpoints; supplement only
    # the neighboring UTC files needed to bracket the requested interval.
    if not gim_cycle or min(gim_cycle) > start:
        load_day(start.normalize() - pd.Timedelta(days=1))
    if not gim_cycle or max(gim_cycle) < end:
        load_day(end.normalize() + pd.Timedelta(days=1))
    if not gim_cycle:
        raise RuntimeError("No GIM files loaded for this cycle")
    return (
        GimMaps(sorted(gim_cycle.items()), interval_seconds=gim_cycle.interval_seconds),
        lat_grid,
        lon_grid,
    )
