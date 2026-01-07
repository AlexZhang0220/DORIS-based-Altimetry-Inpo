from pathlib import Path
import pandas as pd

def load_cycle_table(cycle_csv_path: Path) -> pd.DataFrame:
    """
    Load satellite cycle metadata table and construct start/end datetimes.

    This function reads a cycle definition CSV file (e.g. Jason-3 cycle tables)
    and converts the provided date and time strings into pandas Timestamps.
    """
    df_cyc = pd.read_csv(cycle_csv_path)

    df_cyc["start_datetime"] = pd.to_datetime(
        df_cyc["start_day"] + " " + df_cyc["start_time"]
    )
    df_cyc["end_datetime"] = pd.to_datetime(
        df_cyc["end_date"] + " " + df_cyc["end_time"]
    )

    return df_cyc


def get_cycle_time_window(
    df_cycle: pd.DataFrame,
    cycle_number: int
):
    """
    Extract the start and end time of a given satellite cycle.
    """
    row = df_cycle.loc[df_cycle["cycle_number"] == cycle_number]
    if row.empty:
        raise ValueError(f"cycle_number={cycle_number} not found in cycle table.")

    row = row.iloc[0]
    return row["start_datetime"], row["end_datetime"]