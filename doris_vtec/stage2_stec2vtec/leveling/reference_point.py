import pandas as pd


def select_max_elevation_points(df_pass: pd.DataFrame) -> pd.DataFrame:
    """
    Select one leveling reference point per pass using maximum elevation.

    Parameters
    ----------
    df_pass : pd.DataFrame
        Pass-wise observation DataFrame.
        Must contain:
            - station_code
            - pass_id
            - elevation
            - map_value

    Returns
    -------
    df_ref : pd.DataFrame
        DataFrame containing one reference point per pass.
        The returned DataFrame is a subset of df_pass and
        preserves all original columns.
    """

    required_cols = {"station_code", "pass_id", "elevation", "map_value"}
    missing = required_cols - set(df_pass.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # index of maximum elevation per (station, pass)
    idx = (
        df_pass
        .groupby(["station_code", "pass_id"])["elevation"]
        .idxmax()
    )

    df_ref = df_pass.loc[idx].reset_index(drop=True)

    return df_ref