import pandas as pd
from doris_vtec.io.utils import parse_sinex_time
from pathlib import Path

# ============================================================
# Section extractor
# ============================================================

def extract_sinex_sections(file_path: str) -> dict[str, list[str]]:
    """
    Extract desired sections from SINEX file.
    """
    desired = {
        "SOLUTION/EPOCHS",
        "SITE/ANTENNA",
        "SOLUTION/ESTIMATE",
    }

    sections = {}
    current_section = None
    buffer = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("+"):
                name = line[1:].strip()
                if name in desired:
                    current_section = name
                    buffer = []
                else:
                    current_section = None

            elif line.startswith("-") and current_section:
                sections[current_section] = buffer.copy()
                current_section = None

            elif current_section:
                buffer.append(line)

    return sections


# ============================================================
# Parsers
# ============================================================

def parse_solution_epochs(records: list[str]) -> pd.DataFrame:
    rows = []

    for r in records[1:]:
        rows.append({
            "station_code": r[1:5].strip(),
            "solution_index": int(r[9:13].strip()),
            "start_obs": parse_sinex_time(r[15:28].strip()),
            "end_obs": parse_sinex_time(r[28:41].strip()),
        })

    return pd.DataFrame(rows)


def parse_site_antenna(records: list[str]) -> pd.DataFrame:
    rows = []

    for r in records[1:]:
        rows.append({
            "station_code": r[1:5].strip(),
            "solution_index": int(r[9:13].strip()),
            "antenna_type": r[42:62].strip(),
        })

    return pd.DataFrame(rows)


def parse_solution_estimate(records: list[str]) -> pd.DataFrame:
    rows = []
    i = 1

    while i + 5 < len(records):
        stax, stay, staz, velx, vely, velz = records[i:i+6]

        rows.append({
            "station_code": stax[14:18].strip(),
            "solution_index": int(stax[22:26].strip()),
            "ref_epoch": parse_sinex_time(stax[27:39].strip()),
            "x": float(stax[47:68]),
            "y": float(stay[47:68]),
            "z": float(staz[47:68]),
            "vx": float(velx[47:68]),
            "vy": float(vely[47:68]),
            "vz": float(velz[47:68]),
        })

        i += 6

    return pd.DataFrame(rows)


# ============================================================
# Public API
# ============================================================

def read_sinex_file(file: str) -> pd.DataFrame:
    """
    Read SINEX file and return station coordinates and velocities.

    Units:
        - x, y, z   : meters (m)
        - vx, vy, vz: meters per year (m/year)

    Returns
    -------
    pd.DataFrame
        Columns:
        - station_code
        - solution_index
        - ref_epoch
        - x, y, z        (m)
        - vx, vy, vz     (m/year)
        - start_obs
        - end_obs
        - antenna_type
    """

    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(f"SINEX file not found: {file}")
    
    sections = extract_sinex_sections(file)
    
    df_est = parse_solution_estimate(sections.get("SOLUTION/ESTIMATE", []))
    df_epo = parse_solution_epochs(sections.get("SOLUTION/EPOCHS", []))
    df_ant = parse_site_antenna(sections.get("SITE/ANTENNA", []))

    df = (
        df_est
        .merge(df_epo, on=["station_code", "solution_index"], how="left")
        .merge(df_ant, on=["station_code", "solution_index"], how="left")
    )

    return df
 