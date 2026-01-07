DEFAULT_STAGE2_CONFIG = {
    # ------------------------
    # pass detection
    # ------------------------
    "min_obs_count": 30,   # minimum obs per pass
    "elev_thres": 10.0,    # deg

    # ------------------------
    # reference point
    # ------------------------
    "min_ref_elev": 0.0,  # deg
    # ------------------------
    # leveling
    # ------------------------
    "range_ratio_ja3": 0.925,    # scaling for STEC correction
}