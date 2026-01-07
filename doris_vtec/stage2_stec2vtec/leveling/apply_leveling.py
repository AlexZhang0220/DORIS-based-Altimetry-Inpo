import pandas as pd


def apply_passwise_leveling(
    df_pass: pd.DataFrame,
    df_ref: pd.DataFrame,
    range_ratio: float,
    min_ref_elev: float | None = None,
) -> pd.DataFrame:
    """
    Apply GIM-based pass-wise VTEC leveling.

    Parameters
    ----------
    df_pass : pd.DataFrame
        Pass-wise observations with STEC and mapping function.
    df_ref : pd.DataFrame
        Reference points of gim, must contain:
            - station_code
            - pass_id
            - stec
            - map_value
            - gim_vtec
            - elevation
    range_ratio : float
        Scaling factor applied to GIM-based STEC.
    min_ref_elev : float, optional
        Minimum elevation angle for valid reference points.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected STEC and VTEC.
    """

    df = df_pass.copy()

    # ---- merge reference info into pass df ----
    ref_cols = [
        "station_code",
        "pass_id",
        "stec",
        "map_value",
        "gim_vtec",
        "elevation",
    ]
    df = df.merge(
        df_ref[ref_cols],
        on=["station_code", "pass_id"],
        how="left",
        suffixes=("", "_ref"),
    )

    # ---- optional elevation cutoff ----
    if min_ref_elev is not None:
        valid = df["elevation_ref"] >= min_ref_elev
    else:
        valid = df["gim_vtec"].notna()
    # ---- compute delta STEC per pass ----
    gim_stec_ref = df["gim_vtec"] * df["map_value_ref"]
    delta = gim_stec_ref * range_ratio - df["stec_ref"]

    # ---- apply correction ----
    df.loc[valid, "stec"] = df.loc[valid, "stec"] + delta[valid]
    df.loc[valid, "vtec"] = df.loc[valid, "stec"] / df.loc[valid, "map_value"]

    return df