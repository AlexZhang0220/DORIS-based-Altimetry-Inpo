import numpy as np
import pandas as pd
import constant
from ..geometry.elevation import compute_pco_projection_cosines


def add_doris_stec(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Add DORIS ionospheric observation (dion and STEC) to observation DataFrame.

    The ionospheric delay is derived from dual-frequency DORIS phase
    observations, including geometry-dependent corrections.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Must contain columns:
        - L1, L2
        - sta_x, sta_y, sta_z and sat_x, sat_y, sat_z (meters, ECEF)
        - PRN
        - station_freq_shift
        - sta_ant_type

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - dion    (meters)
        - STEC    (TECU)
    """
    c = constant.c

    # --------------------------------------------------
    # Frequency-dependent wavelength correction
    # --------------------------------------------------
    shift = df_obs["station_freq_shift"].values

    d_f1 = (5e6 * 407.25 * (1 + shift * constant.d_p_multik))
    d_lambda1 = c / d_f1

    d_f2 = (5e6 * 80.25 * (1 + shift * constant.d_p_multik))
    d_lambda2 = c / d_f2

    d_k = (d_f2**2) / (d_f1**2 - d_f2**2)

    d_w = (d_f1**2) / 40.3

    # --------------------------------------------------
    # Raw ionospheric combination
    # --------------------------------------------------
    L1 = df_obs["L1"].values
    L2 = df_obs["L2"].values

    dion = d_k * (L1 * d_lambda1 - L2 * d_lambda2)

    # --------------------------------------------------
    # Geometry-dependent corrections
    # --------------------------------------------------
    prn_map = {
        "L27": constant.d_geomcorrJa2,
        "L39": constant.d_geomcorrJa3,
        "L12": constant.d_geomcorrCr2,
        "L45": constant.d_geomcorrHy2,
        "L46": constant.d_geomcorrSar,
    }
    sat_off = df_obs["prn"].map(prn_map).fillna(0.0).values

    ant_off = np.where(
        df_obs["antenna_type"].str.startswith("STAREC"),
        constant.d_STAREC,
        np.where(
            df_obs["antenna_type"].str.startswith("ALCATEL"),
            constant.d_ALCATEL,
            0.0,
        ),
    )

    satellite_cosine, ground_cosine = compute_pco_projection_cosines(df_obs)
    dion += d_k * (sat_off * satellite_cosine + ant_off * ground_cosine)

    # --------------------------------------------------
    # Convert to STEC (TECU)
    # --------------------------------------------------
    stec = dion * d_w / 1e16

    df_obs["dion"] = dion
    df_obs["stec"] = stec
    df_obs.attrs["pco_projection_model"] = "nominal_los_v1"

    return df_obs
