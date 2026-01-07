import numpy as np
import pandas as pd
import constant


def add_mapping_function(df_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Add ionospheric mapping function value to observation DataFrame.

    A modified single-layer ionospheric model is used.

    Parameters
    ----------
    df_obs : pd.DataFrame
        Must contain column:
        - elevation   (degrees)

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional column:
        - map_value   (dimensionless)
    """
    elevation_rad = np.deg2rad(df_obs["elevation"].values)

    AE84 = constant.AE84
    layer_height = constant.ION_HEIGHT
    alpha = constant.MAPPING_ALPHA

    z_angle = np.pi / 2 - elevation_rad
    ratio = AE84 / (AE84 + layer_height)

    df_obs["map_value"] = 1.0 / np.cos(
        np.arcsin(ratio * np.sin(alpha * z_angle))
    )

    return df_obs