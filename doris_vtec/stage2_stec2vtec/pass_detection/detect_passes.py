import numpy as np
import pandas as pd

from .time_segmentation import split_dataframe_by_time_gap
from .doris_cycle_slip import split_by_cycle_slip


def detect_passes(
    obs_per_station: pd.DataFrame,
    min_obs_count: int,
    elev_thres: float,
) -> pd.DataFrame:
    """
    Detect clean DORIS passes for a single station.

    Parameters
    ----------
    obs_per_station : pd.DataFrame
        Observation DataFrame for ONE station.
    min_obs_count : int
        Minimum number of observations required for a valid pass.
    elev_thres : float
        Minimum elevation angle (deg) for pass detection.
    columns_to_keep : list of str
        Columns preserved in the output pass DataFrame.

    Returns
    -------
    pd.DataFrame
        Concatenated pass DataFrame with a unique pass_id column.
        Returns empty DataFrame if no valid pass is detected.
    """

    # -------------------------------------------------
    # 1. Time-contiguous segmentation (coarse pass split)
    # -------------------------------------------------
    # Missing phases cannot participate in either fit. Filter before time
    # segmentation so a run of missing phases still creates the original gap.
    valid_phase = np.isfinite(
        obs_per_station[["L1", "L2"]].to_numpy(dtype=float)
    ).all(axis=1)
    grouped_obs_list = split_dataframe_by_time_gap(
        obs_per_station.loc[valid_phase].copy(),
        elev_thres=elev_thres,
        min_obs_count=min_obs_count,
    )

    if not grouped_obs_list:
        return pd.DataFrame()

    # -------------------------------------------------
    # 2. Cycle-slip-aware refinement
    # -------------------------------------------------
    all_passes: list[pd.DataFrame] = []
    pass_counter = 1

    for grouped_obs in grouped_obs_list:

        sub_passes = split_by_cycle_slip(
            grouped_obs,
            min_obs_count=min_obs_count,
        )

        for seg in sub_passes:
            seg = seg.copy()
            seg["pass_id"] = pass_counter
            all_passes.append(seg)
            pass_counter += 1

    # -------------------------------------------------
    # 3. Concatenate results
    # -------------------------------------------------
    if all_passes:
        return pd.concat(all_passes, ignore_index=True)
    else:
        return pd.DataFrame()
