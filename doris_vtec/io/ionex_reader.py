import datetime as dt
from pathlib import Path

import numpy as np


class GimMaps(dict):
    """UTC-epoch maps carrying the IONEX declared temporal interval."""

    def __init__(self, *args, interval_seconds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval_seconds = interval_seconds


def read_ionex_file(ionex_file: str):
    """Read the legacy 13-map, 71 x 73 IGS grid; reject other layouts.

    Map epochs are UTC, and the returned three-tuple is unchanged.
    IONEX's missing-value code is removed before applying EXPONENT.
    """
    ionex_file = Path(ionex_file)
    if not ionex_file.is_file():
        raise FileNotFoundError(f"IONEX file not found: {ionex_file}")
    with open(ionex_file, errors="strict") as stream:
        lines = stream.readlines()

    exponent = 0
    header = {}
    for idx, line in enumerate(lines):
        for label in ("# OF MAPS IN FILE", "LAT1 / LAT2 / DLAT", "LON1 / LON2 / DLON", "INTERVAL"):
            if label in line:
                if label == "INTERVAL" and label in header:
                    raise ValueError("Duplicate IONEX INTERVAL declaration")
                header[label] = [float(value) for value in line[:60].split()]
        if "EXPONENT" in line:
            exponent = int(line[:60].split()[0])
        if "END OF HEADER" in line:
            idx += 1
            break
    else:
        raise ValueError(f"IONEX header is incomplete: {ionex_file}")

    interval = header.get("INTERVAL", [])
    if len(interval) != 1 or not np.isfinite(interval[0]) or interval[0] <= 0:
        raise ValueError("Missing or invalid IONEX INTERVAL")

    expected = {
        "# OF MAPS IN FILE": (13,),
        "LAT1 / LAT2 / DLAT": (87.5, -87.5, -2.5),
        "LON1 / LON2 / DLON": (-180.0, 180.0, 5.0),
    }
    for label, values in expected.items():
        if label in header and not np.array_equal(header[label], values):
            raise ValueError(f"Unsupported IONEX {label}: {header[label]}; expected {values}")

    body = lines[idx:]
    if sum("START OF TEC MAP" in line for line in body) != 13:
        raise ValueError("IONEX must contain exactly 13 TEC maps for the legacy IGS reader")
    lat_grid = np.linspace(87.5, -87.5, 71)
    lon_grid = np.linspace(-180.0, 180.0, 73)
    cursor = 0

    def take(label=None):
        nonlocal cursor
        if cursor >= len(body):
            raise ValueError("Incomplete IONEX TEC map")
        line = body[cursor]
        cursor += 1
        if label is not None and label not in line:
            raise ValueError(f"Expected IONEX {label}, got {line.strip()!r}")
        return line

    gim = GimMaps(interval_seconds=interval[0])
    map_height = None
    for _ in range(13):
        take("START OF TEC MAP")
        fields = take("EPOCH OF CURRENT MAP")[:60].split()
        epoch = dt.datetime(*(int(value) for value in fields[:6]))
        tec_map = np.empty((71, 73), dtype=float)
        for ilat, latitude in enumerate(lat_grid):
            row_line = take("LAT/LON1/LON2/DLON/H")
            try:
                row = [float(row_line[2 + 6 * i:8 + 6 * i]) for i in range(5)]
            except ValueError as exc:
                raise ValueError("Malformed fixed-width IONEX map row") from exc
            if len(row) != 5 or not np.array_equal(row[:4], (latitude, -180.0, 180.0, 5.0)):
                raise ValueError(f"Unsupported or inconsistent IONEX map row: {row}")
            if map_height is None:
                map_height = row[4]
            elif row[4] != map_height:
                raise ValueError("IONEX map heights are inconsistent")
            values = []
            for _ in range(5):
                value_line = take().rstrip("\n\r")
                values.extend(float(value_line[j:j + 5]) for j in range(0, len(value_line), 5)
                              if value_line[j:j + 5].strip())
            if len(values) != 73:
                raise ValueError(f"IONEX row has {len(values)} values; expected 73")
            tec_map[ilat] = values
        take("END OF TEC MAP")
        tec_map[tec_map == 9999] = np.nan
        tec_map *= 10.0 ** exponent
        if epoch in gim:
            raise ValueError(f"Duplicate IONEX TEC epoch: {epoch}")
        gim[epoch] = tec_map
    return gim, lat_grid, lon_grid
