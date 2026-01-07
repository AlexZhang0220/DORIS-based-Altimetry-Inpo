import numpy as np
import pandas as pd
from pathlib import Path

def split_rinex_header_obs(
    lines: list[str],
) -> tuple[list[str], list[str]]:
    end_idx = next(
        i for i, line in enumerate(lines)
        if "END OF HEADER" in line
    )
    header_lines = lines[: end_idx + 1]
    obs_lines = lines[end_idx + 1 :]
    return header_lines, obs_lines


# ============================================================
# Header parser
# ============================================================

def parse_rinex_header(header_lines: list[str]) -> dict:
    """
    Parse header information from a DORIS RINEX file.
    """
    obs_type = []
    stations = []
    prn = None

    SATELLITE_NAME_TO_PRN = {
        "JASON-3":     "L39",
        "CRYOSAT-2":   "L12",
        "SARAL":       "L46",
        "SENTINEL-3A": "L74",
        "SENTINEL-3B": "L98",
    }

    for line in header_lines:
        label = line[60:80].strip()

        if label == "SYS / # / OBS TYPES":
            n = int(line[3:6])
            obs_type = [
                line[i+6:i+10].strip()
                for i in range(0, 4 * n, 4)
            ]

        elif label == "STATION REFERENCE":
            stations.append({
                "station_id": int(line[1:3]),
                "station_code": line[5:9].strip(),
                "freq_shift": int(line[52:56]),
                "time_ref": False,
            })

        elif label == "TIME REF STATION":
            sid = int(line[1:3])
            for s in stations:
                if s["station_id"] == sid:
                    s["time_ref"] = True
                    s["time_bias"] = float(line[5:19])   # time beacon reference vs. TAI reference time [1e-6 s]
                    s["time_shift"] = float(line[21:35]) # time beacon reference shift [1e-14 s/s]

        elif label == "SATELLITE NAME":
            name = line[0:60].strip()
            if name in SATELLITE_NAME_TO_PRN:
                prn = SATELLITE_NAME_TO_PRN[name]

        elif label == "END OF HEADER":
            break

    return {
        "obs_type": obs_type,
        "stations": stations,
        "prn": prn,
    }


# ============================================================
# Observation body parser
# ============================================================

def parse_rinex_obs(obs_lines: list[str], header: dict) -> pd.DataFrame:
    """
    Parse observation records from RINEX body.
    """
    records = []
    current_epoch = None
    receiver_clock_offset = None

    obs_type = header["obs_type"]
    prn = header["prn"]
    stations = {
        s["station_id"]: s
        for s in header["stations"]
    }

    for line in obs_lines:
        if line.startswith(">"):
            current_epoch = pd.to_datetime(
                line[2:31],
                format="%Y %m %d %H %M %S.%f"
            )
            receiver_clock_offset = float(line[44:56])
            continue

        if not line.startswith("D"):
            continue

        sid = int(line[1:3])

        obs_fields = [
            float(line[j:j+14].strip())
            if line[j:j+14].strip() else np.nan
            for j in range(3, 83, 16)
        ]

        record = {
            "obs_epoch": current_epoch,
            "receiver_clock_offset": receiver_clock_offset,
            "station_id": sid,
            "station_code": stations[sid]["station_code"],
            "station_freq_shift": stations[sid]["freq_shift"],
            "prn": prn,
        }

        for k in ("L1", "L2", "C1", "C2"):
            record[k] = (
                obs_fields[obs_type.index(k)] * (10.0 if k.startswith("C") else 1.0)
                if k in obs_type else np.nan
            )

        records.append(record)

    return pd.DataFrame.from_records(records)


# ============================================================
# Public API
# ============================================================

def read_rinex_file(file_path: str) -> pd.DataFrame:
    """
    Read DORIS RINEX observation file and return observation DataFrame.

    Units:
        - L1, L2 : cycles
        - C1, C2 : meters (m)

    Returns
    -------
    pd.DataFrame
        Columns:
        - obs_epoch
        - station_id
        - station_code
        - station_freq_shift
        - prn
        - L1, L2          (cycles)
        - C1, C2          (m)
        - receiver_clock_offset
    """

    file = Path(file_path)
    if not file.is_file():
        raise FileNotFoundError(f"RINEX file not found: {file}")

    with open(file, "r") as f:
        lines = f.readlines()

    header_lines, obs_lines = split_rinex_header_obs(lines)

    header = parse_rinex_header(header_lines)

    df_obs = parse_rinex_obs(obs_lines, header)

    return df_obs