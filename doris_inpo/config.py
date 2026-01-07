DEFAULT_INPO_CONFIG = {
    # -------------------------------------------------
    # pass / observation filtering
    # -------------------------------------------------
    "min_obs_count": 30,     # minimum DORIS obs per pass
    "elev_thres": 10.0,      # elevation mask [deg]

    # -------------------------------------------------
    # reference / geometry
    # -------------------------------------------------
    "max_station_distance": 167500.0,  # km

    # -------------------------------------------------
    # spatial interpolation window (adaptive)
    # -------------------------------------------------
    "min_distance": 100.0,   # km
    "max_distance": 1000.0,  # km
    "distance_step": 100.0,  # km

    # -------------------------------------------------
    # leveling / scaling
    # -------------------------------------------------
    "range_ratio_ja3": 0.925,  # STEC → VTEC scaling

    # -------------------------------------------------
    # diagnostics
    # -------------------------------------------------
    "enable_roti": False,
}