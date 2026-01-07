import numpy as np
import datetime as dt
from pathlib import Path


def read_ionex_file(ionex_file: str):
    """
    Read an IONEX (GIM) file and return TEC maps indexed by epoch.

    Returns
    -------
    gim : dict[datetime -> np.ndarray]
        Each value is a 2D TEC map (lat, lon) in TECU.
    lat_grid : np.ndarray
        Latitude grid (degrees).
    lon_grid : np.ndarray
        Longitude grid (degrees).
    """

    ionex_file = Path(ionex_file)
    if not ionex_file.is_file():
        raise FileNotFoundError(f"IONEX file not found: {ionex_file}")

    with open(ionex_file, errors="ignore") as f:
        lines = f.readlines()

    # =========================
    # Parse header
    # =========================
    exponent = 0
    idx = 0

    while True:
        line = lines[idx]

        if "EXPONENT" in line:
            exponent = int(line[0:80].split()[0])

        if "END OF HEADER" in line:
            idx += 1
            break

        idx += 1

    lines = lines[idx:]  # strip header

    # =========================
    # IONEX grid definition (IGS GIM)
    # =========================
    ntime, nlat, nlon = 13, 71, 73

    gim = {}

    # =========================
    # Read TEC maps
    # =========================
    for _ in range(ntime):

        # Start of TEC map
        lines.pop(0)

        # Epoch line
        epoch_line = lines.pop(0).split()
        epoch = dt.datetime(
            int(epoch_line[0]),
            int(epoch_line[1]),
            int(epoch_line[2]),
            int(epoch_line[3]),
            int(epoch_line[4]),
            int(epoch_line[5]),
        )

        tec_map = np.zeros((nlat, nlon), dtype=float)

        for ilat in range(nlat):

            # Latitude header line (lat, lon1, lon2, dlon, height)
            lines.pop(0)

            tec_vals = []
            for _ in range(5):
                tec_vals.extend(lines.pop(0).split())

            # Last longitude (180°) duplicates -180°, drop it
            tec_map[ilat, :] = np.asarray(tec_vals[:], dtype=float)

        # End of TEC map
        lines.pop(0)

        # Apply exponent immediately (physical TECU)
        if exponent != 0:
            tec_map *= 10.0 ** exponent

        gim[epoch] = tec_map

    # =========================
    # Standard IGS grids
    # =========================
    lat_grid = np.linspace(87.5, -87.5, nlat)
    lon_grid = np.linspace(-180, 180 , nlon)

    return gim, lat_grid, lon_grid