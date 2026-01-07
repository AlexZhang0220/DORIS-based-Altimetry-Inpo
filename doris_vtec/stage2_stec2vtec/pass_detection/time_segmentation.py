import pandas as pd


def split_dataframe_by_time_gap(
    df: pd.DataFrame,
    time_col: str = "obs_epoch",
    elev_col: str = "elevation",
    elev_thres: float = 10.0,
    max_gap_seconds: float = 9.0,
    min_obs_count: int = 30,
) -> list[pd.DataFrame]:
    """
    Split observations into time-contiguous segments.

    Returns
    -------
    list of DataFrame
        Each DataFrame is a candidate pass segment.
    """

    df = df.sort_values(by=time_col).reset_index(drop=True)

    time_diff = df[time_col].diff().dt.total_seconds().fillna(0.0)
    segment_id = (time_diff > max_gap_seconds).cumsum()
    df = df.assign(segment_id=segment_id)

    segments = []
    for _, group in df.groupby("segment_id"):
        group = group[group[elev_col] > elev_thres]
        if len(group) >= min_obs_count:
            segments.append(group.reset_index(drop=True))

    return segments