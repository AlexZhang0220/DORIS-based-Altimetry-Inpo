import numpy as np
import pandas as pd
from scipy.stats import zscore

import constant
from ...stage1_obs2stec.geometry.elevation import compute_pco_projection_cosines


def _fit_residuals(X, y):
    """Apply the existing least-squares fit only to finite input and output."""
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite cycle-slip fit input")

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    if not np.isfinite(residuals).all():
        raise ValueError("Non-finite cycle-slip fit residuals")
    return residuals


def _projected_pco_phase_background(grouped_obs, wavelength_l1, wavelength_l2,
                                    ground_l1, ground_l2):
    """Project each endpoint before forming the existing 5L1-L2 background."""
    satellite_cosine, ground_cosine = compute_pco_projection_cosines(grouped_obs)
    return (
        (5.0 * constant.jason3_L1_pco / wavelength_l1
         - constant.jason3_L2_pco / wavelength_l2) * satellite_cosine
        + (5.0 * ground_l1 / wavelength_l1
           - ground_l2 / wavelength_l2) * ground_cosine
    )


def split_by_cycle_slip(
    grouped_obs: pd.DataFrame,
    min_obs_count: int,
) -> list[pd.DataFrame]:
    """
    Refine a time-contiguous segment using cycle-slip detection.

    Parameters
    ----------
    grouped_obs : pd.DataFrame
        Time-contiguous observation segment for a single station.
    min_obs_count : int
        Minimum number of observations required for a valid pass.

    Returns
    -------
    list of pd.DataFrame
        Clean sub-pass segments after cycle-slip refinement.
    """

    pass_per_station = []

    # -------------------------------------------------
    # Instrument / PRN check
    # -------------------------------------------------
    sta_ant_type = grouped_obs["antenna_type"].values[0]
    prn = grouped_obs["prn"].values[0]

    if sta_ant_type[0:6] == "STAREC" and prn == "L39":
        ground_l1 = constant.starec_L1_pco
        ground_l2 = constant.starec_L2_pco
    elif sta_ant_type[0:7] == "ALCATEL" and prn == "L39":
        ground_l1 = constant.alcatel_L1_pco
        ground_l2 = constant.alcatel_L2_pco
    else:
        return []

    # -------------------------------------------------
    # Frequency / wavelength definition
    # -------------------------------------------------
    shift = grouped_obs["station_freq_shift"].values[0]
    c = constant.c

    d_f1 = 5e6 * 407.25 * (1 + shift * constant.d_p_multik)
    d_f2 = 5e6 * 80.25  * (1 + shift * constant.d_p_multik)

    d_lambda1 = c / d_f1
    d_lambda2 = c / d_f2

    cycle_slip_pco_background = _projected_pco_phase_background(
        grouped_obs, d_lambda1, d_lambda2, ground_l1, ground_l2
    )
    geom_factor = 5 / d_lambda1 - 1 / d_lambda2

    # -------------------------------------------------
    # Skip large jump cases 
    # -------------------------------------------------
    if (
        grouped_obs["L1"].diff().abs().max(skipna=True) > 5e5
        and len(grouped_obs["obs_epoch"].dt.date.unique()) == 1
    ):
        return []

    # -------------------------------------------------
    # Build cycle-slip sensitive observable
    # -------------------------------------------------
    elev_rad = np.deg2rad(grouped_obs["elevation"].values)
    relative_sec = (
        grouped_obs["obs_epoch"] - grouped_obs["obs_epoch"].iloc[0]
    ).dt.total_seconds()

    cycle_slip_phase = 5 * grouped_obs["L1"] - grouped_obs["L2"]

    geom_distances = np.sqrt(
        (grouped_obs["sat_x"] - grouped_obs["sta_x"]) ** 2
        + (grouped_obs["sat_y"] - grouped_obs["sta_y"]) ** 2
        + (grouped_obs["sat_z"] - grouped_obs["sta_z"]) ** 2
    )

    corrected_geom_distances = (
        geom_factor * geom_distances
        - cycle_slip_pco_background
    )

    mapping_func = 1.0 / np.sin(elev_rad)
    estimated_cycle_slip = cycle_slip_phase - corrected_geom_distances

    # -------------------------------------------------
    # First-stage residual analysis
    # -------------------------------------------------
    X = np.column_stack((
        np.ones(len(relative_sec)),
        relative_sec,
        mapping_func,
    ))
    residuals = _fit_residuals(X, estimated_cycle_slip)
    diff_residuals = np.diff(residuals)

    if np.max(np.abs(diff_residuals)) > 10:
        z_scores = np.abs(zscore(diff_residuals))
        split_points = np.where(z_scores > 3)[0] + 1
    else:
        split_points = np.array([], dtype=int)

    split_points = np.hstack(([0], split_points, [len(estimated_cycle_slip)]))

    # -------------------------------------------------
    # Second-stage refinement
    # -------------------------------------------------
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]

        if end - start < min_obs_count:
            continue

        X_sub = np.column_stack((
            np.ones(end - start),
            relative_sec[start:end],
            mapping_func[start:end],
        ))
        y_sub = estimated_cycle_slip[start:end]

        residuals_sub = _fit_residuals(X_sub, y_sub)
        diff_res_sub = np.diff(residuals_sub)

        jump_idx = np.where(np.abs(diff_res_sub) > 3.5)[0] + 1
        jump_idx = np.hstack(([0], jump_idx, [len(y_sub)]))

        for j in range(len(jump_idx) - 1):
            seg_start = start + jump_idx[j]
            seg_end   = start + jump_idx[j + 1]

            if seg_end - seg_start >= min_obs_count:
                seg = grouped_obs.iloc[seg_start:seg_end].copy()
                pass_per_station.append(seg)

    return pass_per_station
