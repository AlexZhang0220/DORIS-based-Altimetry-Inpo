import numpy as np

def idw(dist_km, values):
    """
    Inverse Distance Weighting (IDW).
    """
    dist = np.asarray(dist_km, dtype=float)
    vals = np.asarray(values, dtype=float)

    finite = np.isfinite(dist) & np.isfinite(vals)
    zero = finite & (dist == 0)
    if np.any(zero):
        return float(np.mean(vals[zero]))

    mask = finite & (dist > 0)
    if not np.any(mask):
        return np.nan

    w = 1.0 / dist[mask]
    return np.sum(w * vals[mask]) / np.sum(w)
