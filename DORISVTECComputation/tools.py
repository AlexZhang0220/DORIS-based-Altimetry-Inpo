import numpy as np
import pandas as pd
from pathlib import Path
import constant
from scipy.interpolate import RegularGridInterpolator
from numba import njit

@njit
def batch_lagrange_interp_1d(x_known, y_known, x_interp):
    n = len(x_known)
    result = np.zeros_like(x_interp)
    for j in range(len(x_interp)):
        x = x_interp[j]
        total = 0.0
        for i in range(n):
            term = y_known[i]
            for k in range(n):
                if k != i:
                    term *= (x - x_known[k]) / (x_known[i] - x_known[k])
            total += term
        result[j] = total
    return result

def get_igs_vtec(GIMVTEC: list[np.ndarray], df_max: pd.DataFrame) -> np.ndarray:
    MaxLatIndex, MaxLonIndex = 71, 73

    inpo_funcs = [
        RegularGridInterpolator((range(MaxLatIndex), range(MaxLonIndex)), GIMVTEC[i], method="linear")
        for i in range(13)
    ]

    mapped_lon = (df_max['ipp_lon'] + 180) % 360 - 180
    lat_idx = (df_max['ipp_lat'] + 87.5) / 2.5
    lon_idx = (mapped_lon + 180) / 5

    epoch = pd.to_datetime(df_max['obs_epoch'])
    t_idx = epoch.dt.hour // 2
    t_scale = ((epoch.dt.hour + epoch.dt.minute / 60 + epoch.dt.second / 3600) % 2) / 2

    coords = np.stack([lat_idx, lon_idx], axis=1)

    vtec1 = np.array([inpo_funcs[i]([pt])[0] for i, pt in zip(t_idx, coords)])
    vtec2 = np.array([inpo_funcs[i+1]([pt])[0] for i, pt in zip(t_idx, coords)])
    
    return (1 - t_scale) * vtec1 + t_scale * vtec2

def haversine_vec(lat1, lon1, lat2_array, lon2_array, R=constant.AE84+506700):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_array)
    dphi = np.radians(lat2_array - lat1)
    dlambda = np.radians(lon2_array - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def idw(distances, values, power=2):

    distances = np.asarray(distances)
    values = np.asarray(values)
    if np.any(distances == 0):
        return values[distances == 0][0]

    weights = 1 / distances**power
    weighted_sum = np.sum(weights * values)
    sum_weights = np.sum(weights)

    return weighted_sum / sum_weights

def get_gim_filename(gim_dir: Path, year: int, doy: int) -> str:

    gim_dir_year = gim_dir / f"{year}"
    file_list = [f.name for f in gim_dir_year.iterdir() if f.is_file()]
    inx_file = next((f for f in file_list if f.startswith(f"IGS0OPSFIN_{year}{doy:03d}") and f.endswith("GIM.INX")), None)
    igsg_file = next((f for f in file_list if f.startswith(f"igsg{doy:03d}0.{str(year % 100).zfill(2)}i")), None)
    ion_file = inx_file or igsg_file

    return gim_dir_year / ion_file