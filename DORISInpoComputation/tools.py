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

def get_igs_vtec(GIMVTEC: list[np.ndarray], df_inpo: pd.DataFrame) -> np.ndarray:
    # obtain interpolation results for corresponding time and ipp position at df_inpo

    MaxLatIndex, MaxLonIndex = 71, 73
    inpo_funcs = [
        RegularGridInterpolator((range(MaxLatIndex), range(MaxLonIndex)), GIMVTEC[i], method="linear")
        for i in range(13)
    ]

    mapped_lon = (df_inpo['ipp_lon'] + 180) % 360 - 180
    lat_idx = (df_inpo['ipp_lat'] + 87.5) / 2.5
    lon_idx = (mapped_lon + 180) / 5

    epoch = pd.to_datetime(df_inpo['obs_epoch'])
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

def get_gim_filename(gim_dir: Path, year: int, doy: int) -> Path:

    gim_dir_year = gim_dir / f"{year}"
    file_list = [f.name for f in gim_dir_year.iterdir() if f.is_file()]
    inx_file = next((f for f in file_list if f.startswith(f"IGS0OPSFIN_{year}{doy:03d}") and f.endswith("GIM.INX")), None)
    igsg_file = next((f for f in file_list if f.startswith(f"igsg{doy:03d}0.{str(year % 100).zfill(2)}i")), None)
    ion_file = inx_file or igsg_file

    return gim_dir_year / ion_file

def latlon_to_unitvec(lat_deg, lon_deg):
    """Convert lat/lon (deg) to unit vectors on the sphere."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    clat = np.cos(lat)
    x = clat * np.cos(lon)
    y = clat * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack((x, y, z))  # shape (N, 3)

def _xy_tangent_km(lat0, lon0, lats, lons):
    """将经纬度相对目标点转换到切平面坐标 (x: 东向 km, y: 北向 km)"""

    def _wrap_deg(x):
        return (x + 180.0) % 360.0 - 180.0
    dlat = np.radians((np.asarray(lats) - lat0))
    dlon = np.radians(_wrap_deg(np.asarray(lons) - lon0))
    x = (constant.AE84 + constant.ion_height) / 1000 * np.cos(np.radians(lat0)) * dlon
    y = (constant.AE84 + constant.ion_height) / 1000 * dlat
    return x, y


def _pca_main_dir(x, y):
    X = np.column_stack([x - np.mean(x), y - np.mean(y)])
    if np.allclose(X, 0.0):  # degenerate
        return np.array([1.0,0.0]), 1.0
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    main = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    ratio = S[0] / max(S[1], 1e-12)
    return main, float(ratio)

def _angle_to_axis(x, y, axis_vec):
    v = np.column_stack([x, y])
    vnorm = np.linalg.norm(v, axis=1) + 1e-12
    axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-12)
    cos_th = (v @ axis) / vnorm
    return np.arccos(np.clip(cos_th, -1, 1))

def _mad_sigma(res):
    mad = np.median(np.abs(res - np.median(res))) + 1e-12
    return 1.4826 * mad

def _huber_w(std_res, c=1.7):
    a = np.abs(std_res)
    w = np.ones_like(a)
    m = a > c
    w[m] = c / a[m]
    return w

# --------------- main function ---------------
def predict_alt_vtec_doris(
    lat0, lon0, t0,
    doris_lats, doris_lons, doris_times, doris_values,
    *,
    # weighting
    ell_km=500.0, tau_h=1.0, alpha_dir=2.5,
    station=None, pass_id=None, quality=None,
    dir_auto_scale=True, dir_alpha_bounds=(1.0, 4.0),
    # regression
    use_time_term=True, min_time_span_min=15.0,
    iter_robust=2, ridge_rel=1e-3
):
    """
    直接输出 Altimetry 目标点 VTEC 预测值（加权局部一阶平面）。
    参数：
      doris_values: 可为 DORIS 绝对 VTEC（则直接输出绝对预测），
                    也可为 DORIS-GIM 残差（则外部自行再加回 GIM 背景）。
    返回：
      vtec_pred, sigma_pred0, beta, diagnostics
    """
    # arrays
    lats = np.asarray(doris_lats, float)
    lons = np.asarray(doris_lons, float)
    vals = np.asarray(doris_values, float)
    t0p = pd.to_datetime(t0)
    t_arr = pd.to_datetime(doris_times)
    dt_h = (t_arr - t0p).to_numpy().astype('timedelta64[s]').astype(float) / 3600.0

    # local geometry
    x, y = _xy_tangent_km(lat0, lon0, lats, lons)
    d_km = np.sqrt(x**2 + y**2)

    # base weights: space/time
    w_space = np.exp(-(d_km**2) / (2.0 * ell_km**2))
    w_time  = np.exp(-(dt_h**2) / (2.0 * tau_h**2))

    # PCA direction weight
    # main_vec, anis_ratio = _pca_main_dir(x, y)
    # if dir_auto_scale:
    #     r = anis_ratio
    #     a_min, a_max = dir_alpha_bounds
    #     r_lo, r_hi = 1.0, 25.0
    #     alpha_eff = a_min + (a_max - a_min) * np.clip((r - r_lo)/(r_hi - r_lo), 0.0, 1.0)
    # else:
    #     alpha_eff = float(alpha_dir)
    # theta = _angle_to_axis(x, y, main_vec)
    # w_dir = np.exp(-alpha_eff * (np.sin(theta)**2))

    # redundancy weight (by station OR pass)
    if station is not None:
        s = pd.Series(np.asarray(station))
        counts = s.map(s.value_counts()).to_numpy()
        w_rep = 1.0 / np.sqrt(counts)
    elif pass_id is not None:
        p = pd.Series(np.asarray(pass_id))
        counts = p.map(p.value_counts()).to_numpy()
        w_rep = 1.0 / np.sqrt(counts)
    else:
        w_rep = np.ones_like(w_space)

    # quality weight (optional)
    if quality is not None:
        q = np.asarray(quality, float)
        s = np.nanmedian(np.abs(q - np.nanmedian(q))) + 1e-12
        w_q = np.exp(-np.clip(q, 0, None) / (3*s))
    else:
        w_q = np.ones_like(w_space)

    # total weights
    w = w_space * w_time * w_rep * w_q
    w = np.clip(w, 1e-12, None)
    w = np.ones_like(w_space)
    # design matrix A = [1, x, y, (t)]
    cols = [np.ones_like(x), x, y]
    use_t = bool(use_time_term)
    if use_t and (np.ptp(dt_h) * 60.0 < min_time_span_min):
        use_t = False
    if use_t:
        cols.append(dt_h)
    A = np.column_stack(cols)
    yv = vals
    n, p = A.shape

    # solve weighted LS with light ridge
    def _solve(A, w, y, ridge_rel):
        Wsqrt = np.sqrt(w)
        Aw = A * Wsqrt[:, None]
        yw = y * Wsqrt
        AtWA = Aw.T @ Aw
        tr = np.trace(AtWA)
        lam = ridge_rel * tr / p if tr > 0 else 0.0
        beta = np.linalg.solve(AtWA, Aw.T @ yw)
        return beta, AtWA, lam

    beta, AtWA, lam = _solve(A, w, yv, ridge_rel)

    # robust (Huber) reweighting
    for _ in range(max(0, int(iter_robust))):
        res = yv - A @ beta
        sigma = _mad_sigma(res)
        if sigma <= 0: break
        w_h = _huber_w(res / sigma, c=1.7)
        w = np.clip(w * w_h, 1e-12, None)
        beta, AtWA, lam = _solve(A, w, yv, ridge_rel)

    # prediction & uncertainty (β0)
    res = yv - A @ beta
    dof_eff = max(np.sum(w) - p, 1.0)
    sigma_loc2 = float((w * res**2).sum() / dof_eff)

    Wsqrt = np.sqrt(w)
    Aw = A * Wsqrt[:, None]
    AtWA = Aw.T @ Aw
    tr = np.trace(AtWA)
    lam = ridge_rel * tr / p if tr > 0 else 0.0
    cov_beta = np.linalg.inv(AtWA + lam*np.eye(p)) * sigma_loc2

    vtec_pred = float(beta[0])                  # 目标点 VTEC（若输入为残差，请自行 +GIM）
    sigma_pred0 = float(np.sqrt(cov_beta[0,0])) # β0 的标准差

    # diagnostics = {
    #     "n_points": int(n),
    #     "used_time_term": bool(use_t),
    #     "anisotropy_ratio": float(anis_ratio),
    #     "alpha_effective": float(alpha_eff),
    #     "ridge_lambda": float(lam),
    #     "cond_AtWA": float(np.linalg.cond(AtWA + lam*np.eye(p))),
    #     "sigma_local": float(np.sqrt(sigma_loc2)),
    #     "weights_sum": float(np.sum(w)),
    # }
    return vtec_pred, sigma_pred0, beta

def fit_local_plane_basic(
    lat0, lon0, t0,
    lats, lons, times, residuals,
    use_time_term=True,
    min_time_span_min=15.0,
    ridge=0.0,
):
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    residuals = np.asarray(residuals, float)

    # 相对时间（小时，带符号）
    t0p = pd.to_datetime(t0)
    t_arr = pd.to_datetime(times)
    dt_h = (t_arr - t0p).to_numpy().astype('timedelta64[s]').astype(float) / 3600.0

    # 切平面坐标（km）
    x, y = _xy_tangent_km(lat0, lon0, lats, lons)

    # 设计矩阵 A
    cols = [np.ones_like(x), x, y]
    use_t = bool(use_time_term)
    if use_t and (np.ptp(dt_h) * 60.0 < min_time_span_min):
        use_t = False
    if use_t:
        cols.append(dt_h)
    A = np.column_stack(cols)  # (n, p)
    y_vec = residuals
    n, p = A.shape

    if n < p:
        raise ValueError(f"邻域点数不足以拟合平面：n={n}, p={p}")

    # 普通最小二乘（可选轻度岭正则）
    AtA = A.T @ A
    Aty = A.T @ y_vec
    if ridge and ridge > 0.0:
        AtA = AtA + ridge * np.eye(p)
    beta = np.linalg.solve(AtA, Aty)

    # 残差与方差
    res = y_vec - A @ beta
    dof = max(n - p, 1)
    sigma_r2 = float((res @ res) / dof)

    # β 协方差 ≈ (A^T A)^(-1) * σ_r^2
    cov_beta = np.linalg.inv(AtA) * sigma_r2
    r_hat = float(beta[0])
    sigma_r0 = float(np.sqrt(cov_beta[0, 0]))

    diag = {
        "n_points": int(n),
        "used_time_term": bool(use_t),
        "cond_AtA": float(np.linalg.cond(AtA)),
        "sigma_local": float(np.sqrt(sigma_r2)),
    }
    return r_hat, sigma_r0, beta, diag