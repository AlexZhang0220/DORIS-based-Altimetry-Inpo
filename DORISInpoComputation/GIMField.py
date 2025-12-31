import numpy as np
import pandas as pd
import constant

# IONEX (IGS GIM) 标准网格尺寸（-87.5:2.5:87.5 共 71；-180:5:180 共 73）
MAX_LAT_IDX, MAX_LON_IDX = 71, 73
LAT_GRID = -87.5 + 2.5 * np.arange(MAX_LAT_IDX)          # shape (71,)
LON_GRID = -180.0 + 5.0 * np.arange(MAX_LON_IDX)         # shape (73,)

def _wrap180(lon_deg: np.ndarray) -> np.ndarray:
    return (lon_deg + 180.0) % 360.0 - 180.0

def _time_bracket_and_weight(t0: pd.Timestamp, times_sorted: list[pd.Timestamp]):
    """找到 t0 所在的时间片 [t1,t2] 及线性权重 w in [0,1]（t0=t1→0, t0=t2→1）。超界贴边。"""
    if t0 <= times_sorted[0]:
        return 0, 0, 0.0
    if t0 >= times_sorted[-1]:
        m1 = len(times_sorted) - 1
        return m1, m1, 0.0
    # 第一个 >= t0 的索引
    j2 = next(j for j, tt in enumerate(times_sorted) if tt >= t0)
    j1 = j2 - 1
    t1, t2 = times_sorted[j1], times_sorted[j2]
    dt = (t2 - t1).total_seconds()
    w = 0.0 if dt <= 0 else (t0 - t1).total_seconds() / dt
    return j1, j2, float(np.clip(w, 0.0, 1.0))

def _nearest_grid_indices(lat_deg: float, lon_deg: float):
    """返回最接近经纬度的栅格索引 (i_lat, j_lon)。经度按 [-180,180) 处理。"""
    lon_wrapped = float(_wrap180(np.array([lon_deg]))[0])
    i = int(np.round((lat_deg + 87.5) / 2.5))     # 近邻
    j = int(np.round((lon_wrapped + 180.0) / 5.0))
    i = int(np.clip(i, 0, MAX_LAT_IDX - 1))
    j = int(np.mod(j, MAX_LON_IDX))               # 经度环绕
    return i, j

def _block_indices(i_center: int, j_center: int, k_lat: int, k_lon: int):
    """给定中心索引与块大小，返回该块的纬度、经度索引数组（纬度夹持，经度环绕）。"""
    h_lat = k_lat // 2
    h_lon = k_lon // 2
    i_start = max(0, i_center - h_lat)
    i_end   = min(MAX_LAT_IDX, i_center + h_lat + 1)
    # 若靠边导致块不满，向另一侧补齐
    if i_end - i_start < k_lat:
        if i_start == 0:
            i_end = min(MAX_LAT_IDX, i_start + k_lat)
        else:
            i_start = max(0, i_end - k_lat)
    i_idx = np.arange(i_start, i_end)                               # 长度 k_lat

    j_start = j_center - h_lon
    j_idx = (j_start + np.arange(k_lon)) % MAX_LON_IDX              # 长度 k_lon（环绕）
    return i_idx, j_idx

def get_igs_vtec_blocks(
    GIMVTEC: list[np.ndarray],
    GIM_TIMES: list[pd.Timestamp],
    df_inpo: pd.DataFrame,
    *,
    block_lat: int = 5,
    block_lon: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据 df_inpo（含 'ipp_lat', 'ipp_lon', 'obs_epoch'），为每一行返回
    目标点附近 block_lat × block_lon 的 GIM 值块（已做时间线性插值到该时刻），
    以及对应的 lat、lon 块坐标。

    返回:
      V_blocks: (N, block_lat, block_lon)  — VTEC(TECU)
      LAT_blocks: (N, block_lat, block_lon)
      LON_blocks: (N, block_lat, block_lon)
    说明:
      - GIMVTEC[t] 的形状需为 (71, 73)，且已乘以 IONEX EXPONENT（单位 TECU）。
      - GIM_TIMES 与 GIMVTEC 一一对应并按时间升序。
      - 经度自动按 [-180,180) 处理，块的经度沿环向连续。
    """
    assert len(GIMVTEC) == len(GIM_TIMES) and len(GIMVTEC) >= 1
    # 预处理输入
    lats = df_inpo["ipp_lat"].to_numpy(float)
    lons = df_inpo["ipp_lon"].to_numpy(float)
    epochs = pd.to_datetime(df_inpo["obs_epoch"]).tolist()
    N = len(lats)

    V_blocks = np.full((N, block_lat, block_lon), np.nan, dtype=float)
    LAT_blocks = np.full((N, block_lat, block_lon), np.nan, dtype=float)
    LON_blocks = np.full((N, block_lat, block_lon), np.nan, dtype=float)

    # 主循环：逐点提取块 + 时间插值
    for k in range(N):
        lat0 = float(lats[k])
        lon0 = float(lons[k])
        t0 = epochs[k]

        # 时间片与权重
        j1, j2, w = _time_bracket_and_weight(t0, GIM_TIMES)
        grid1 = GIMVTEC[j1]
        grid2 = GIMVTEC[j2]

        # 中心栅格 + 块索引
        ic, jc = _nearest_grid_indices(lat0, lon0)
        i_idx, j_idx = _block_indices(ic, jc, block_lat, block_lon)

        # 取两时刻的块并线性插值到 t0
        V1 = grid1[np.ix_(i_idx, j_idx)]                   # (block_lat, block_lon)
        if j2 == j1:
            Vt = V1
        else:
            V2 = grid2[np.ix_(i_idx, j_idx)]
            Vt = (1.0 - w) * V1 + w * V2

        # 对应的经纬度块（经度需沿环向与 j_idx 一致）
        lat_blk = LAT_GRID[i_idx][:, None].repeat(block_lon, axis=1)
        lon_blk = LON_GRID[j_idx][None, :].repeat(block_lat, axis=0)

        V_blocks[k] = Vt
        LAT_blocks[k] = lat_blk
        LON_blocks[k] = lon_blk

    return V_blocks, LAT_blocks, LON_blocks

DEG = np.pi / 180.0
R_EARTH_KM = (constant.AE84 + constant.ion_height) / 1000.0

def _xy_km(lat0, lon0, lats, lons):
    """以 (lat0,lon0) 为原点，将经纬度转为切平面坐标 x(东向km), y(北向km)。"""
    dlat = (lats - lat0) * DEG
    dlon = _wrap180(lons - lon0) * DEG
    x = R_EARTH_KM * np.cos(lat0 * DEG) * dlon
    y = R_EARTH_KM * dlat
    return x, y

def build_gim_field_single_block_quadratic(
    V_block: np.ndarray,
    LAT_block: np.ndarray,
    LON_block: np.ndarray,
    lat0: float,
    lon0: float,
    *,
    ridge_linear: float = 1e-10,
    ridge_quadr: float = 1e-8,
    return_gradients: bool = True,
    return_diagnostics: bool = True
):
    """
    对单个 GIM 小块 (Klat×Klon) 拟合二阶平面:
        V ≈ β0 + βx*x + βy*y + βxx*x² + βyy*y² + βxy*x*y

    参数
    ----
    V_block, LAT_block, LON_block : 2D 数组 (Klat×Klon)
        该 altimetry 点周围的 GIM 小块数据。
    lat0, lon0 : float
        目标点经纬度。
    ridge_linear, ridge_quadr : float
        岭正则强度。
    return_gradients, return_diagnostics : bool
        是否返回梯度与诊断。

    返回
    ----
    beta : (6,)
        二阶系数 [β0, βx, βy, βxx, βyy, βxy]
    grad_xy : (2,) or None
        一阶梯度 (βx, βy)
    diag : dict or None
        诊断信息 {rms, n_points, cond}
    """

    # 拉平数据
    V = np.ravel(V_block).astype(float)
    lat_blk = np.ravel(LAT_block).astype(float)
    lon_blk = np.ravel(LON_block).astype(float)

    m = np.isfinite(V) & np.isfinite(lat_blk) & np.isfinite(lon_blk)
    if m.sum() < 6:
        raise ValueError("Not enough valid points for quadratic fit (need ≥6).")

    # 构造局部切平面坐标 (km)
    x, y = _xy_km(lat0, lon0, lat_blk[m], lon_blk[m])
    z = V[m]

    # 构造设计矩阵 [1, x, y, x², y², x*y]
    A = np.column_stack([np.ones_like(x), x, y, x**2, y**2, x*y])

    # 不同阶次分别加岭正则
    lam = np.array([0.0, ridge_linear, ridge_linear,
                    ridge_quadr, ridge_quadr, ridge_quadr], float)
    Reg = np.diag(lam)

    AtA = A.T @ A + Reg
    Aty = A.T @ z
    try:
        beta = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(AtA, Aty, rcond=None)[0]

    grad_xy = (beta[1], beta[2]) if return_gradients else None

    diag = None
    if return_diagnostics:
        res = z - A @ beta
        rms = float(np.sqrt(np.mean(res**2)))
        condnum = float(np.linalg.cond(AtA))
        diag = {"rms": rms, "n_points": int(m.sum()), "cond": condnum}

    return beta

def build_gim_field_single_block_linear(
    V_block: np.ndarray,
    LAT_block: np.ndarray,
    LON_block: np.ndarray,
    lat0: float,
    lon0: float,
    *,
    ridge_linear: float = 1e-10,   # 一阶项的岭正则
    ridge_bias: float = 0.0,       # 常数项的岭正则（一般保持 0）
    return_gradients: bool = True,
    return_diagnostics: bool = True
):
    """
    对单个 GIM 小块 (Klat×Klon) 拟合一阶平面:
        V ≈ β0 + βx*x + βy*y
    其中 (x, y) 为以 (lat0, lon0) 为切点的局部平面坐标（单位：km）。

    参数
    ----
    V_block, LAT_block, LON_block : 2D 数组 (Klat×Klon)
    lat0, lon0 : float
    ridge_linear, ridge_bias : float
        岭正则强度（分别作用于一阶项与常数项）。
    return_gradients, return_diagnostics : bool

    返回
    ----
    beta : (3,)
        一阶系数 [β0, βx, βy]
    grad_xy : (2,) or None
        一阶梯度 (βx, βy)
    diag : dict or None
        诊断信息:
          - rms: 残差RMS
          - sigma0: 单位权中误差（精度评定指标，推荐使用）
          - pred_se_center: 目标点预测标准差（≈ √Cov(β0,β0)）
          - se_grad: (se_bx, se_by)，梯度标准误
          - n_points: 有效点数
          - cond: 正则化后法矩阵条件数
    """
    # 拉平数据
    V = np.ravel(V_block).astype(float)
    lat_blk = np.ravel(LAT_block).astype(float)
    lon_blk = np.ravel(LON_block).astype(float)

    m = np.isfinite(V) & np.isfinite(lat_blk) & np.isfinite(lon_blk)
    if m.sum() < 3:
        raise ValueError("Not enough valid points for linear fit (need ≥3).")

    # 构造局部切平面坐标 (km)
    x, y = _xy_km(lat0, lon0, lat_blk[m], lon_blk[m])
    z = V[m]

    # 设计矩阵 [1, x, y]
    A = np.column_stack([np.ones_like(x), x, y])

    # 岭正则: diag([ridge_bias, ridge_linear, ridge_linear])
    Reg = np.diag([float(ridge_bias), float(ridge_linear), float(ridge_linear)])

    AtA = A.T @ A + Reg
    Aty = A.T @ z
    try:
        beta = np.linalg.solve(AtA, Aty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(AtA, Aty, rcond=None)[0]

    grad_xy = (beta[1], beta[2]) if return_gradients else None

    diag = None
    if return_diagnostics:
        res = z - A @ beta
        rms = float(np.sqrt(np.mean(res**2)))
        n, p = A.shape
        if n > p:
            # 单位权中误差（精度评定指标，推荐使用）
            sigma0 = float(np.sqrt(np.sum(res**2) / (n - p)))
        else:
            sigma0 = float("nan")

        # 近似参数协方差（含岭正则时为近似）
        try:
            AtA_inv = np.linalg.inv(AtA)
        except np.linalg.LinAlgError:
            AtA_inv = np.linalg.pinv(AtA)

        cov_beta = (sigma0**2) * AtA_inv if np.isfinite(sigma0) else np.full_like(AtA_inv, np.nan)

        # 在目标点 (x=0,y=0) 的预测标准差：只与 β0 的方差有关
        pred_se_center = float(np.sqrt(max(cov_beta[0, 0], 0.0))) if np.isfinite(cov_beta[0, 0]) else float("nan")

        # 梯度标准误
        se_bx = float(np.sqrt(max(cov_beta[1, 1], 0.0))) if np.isfinite(cov_beta[1, 1]) else float("nan")
        se_by = float(np.sqrt(max(cov_beta[2, 2], 0.0))) if np.isfinite(cov_beta[2, 2]) else float("nan")

        condnum = float(np.linalg.cond(AtA))
        diag = {
            "rms": rms,
            "sigma0": sigma0,                 # ✅ 精度评定指标（推荐）
            "pred_se_center": pred_se_center, # 可选精度指标：目标点预测标准差
            "se_grad": (se_bx, se_by),
            "n_points": int(m.sum()),
            "cond": condnum,
        }

    return beta, grad_xy, diag

def build_gim_field_single_block_linear_block(
    V_stack: np.ndarray,     # (N, Klat, Klon)
    LAT_stack: np.ndarray,   # (N, Klat, Klon)
    LON_stack: np.ndarray,   # (N, Klat, Klon)
    lat0,                    # float 或 (N,)
    lon0,                    # float 或 (N,)
    *,
    ridge_linear: float = 1e-10,
    ridge_bias: float = 0.0,
    return_gradients: bool = True,
    return_diagnostics: bool = True,
):
    """
    对 V_stack 的第一个维度循环。每个切片 (Klat, Klon) 进行一阶平面拟合：
        V ≈ β0 + βx*x + βy*y
    返回按 N 汇总的参数、梯度与诊断。

    输出
    ----
    betas: (N, 3)  -> [β0, βx, βy]
    grads: (N, 2) or None  -> (βx, βy)
    diag: dict or None:
        - rms: (N,)
        - sigma0: (N,)             # ✅ 精度评定指标（单位权中误差）
        - pred_se_center: (N,)     # 目标点预测标准差
        - se_grad: (N, 2)          # 梯度标准误
        - n_points: (N,) int
        - cond: (N,)
        - ok: (N,) bool            # 本次拟合是否成功
    """
    assert V_stack.shape == LAT_stack.shape == LON_stack.shape, "shape mismatch"
    N = V_stack.shape[0]

    # 规范化 lat0/lon0：支持标量或 (N,)
    lat0_arr = np.full(N, float(lat0)) if np.isscalar(lat0) else np.asarray(lat0, dtype=float)
    lon0_arr = np.full(N, float(lon0)) if np.isscalar(lon0) else np.asarray(lon0, dtype=float)
    if lat0_arr.shape[0] != N or lon0_arr.shape[0] != N:
        raise ValueError("lat0/lon0 must be scalar or length-N arrays.")

    # 结果容器
    betas = np.full((N, 3), np.nan, dtype=float)
    grads = np.full((N, 2), np.nan, dtype=float) if return_gradients else None

    diag = None
    if return_diagnostics:
        diag = {
            "rms": np.full(N, np.nan, dtype=float),
            "sigma0": np.full(N, np.nan, dtype=float),          # ✅ 精度指标
            "pred_se_center": np.full(N, np.nan, dtype=float),
            "se_grad": np.full((N, 2), np.nan, dtype=float),
            "n_points": np.zeros(N, dtype=int),
            "cond": np.full(N, np.nan, dtype=float),
            "ok": np.zeros(N, dtype=bool),
        }

    for i in range(N):
        V_block = V_stack[i]
        LAT_block = LAT_stack[i]
        LON_block = LON_stack[i]
        try:
            beta, grad_xy, diag_i = build_gim_field_single_block_linear(
                V_block, LAT_block, LON_block,
                lat0_arr[i], lon0_arr[i],
                ridge_linear=ridge_linear,
                ridge_bias=ridge_bias,
                return_gradients=return_gradients,
                return_diagnostics=return_diagnostics
            )
            betas[i, :] = beta
            if return_gradients:
                grads[i, :] = grad_xy

            if return_diagnostics and diag_i is not None:
                diag["rms"][i] = diag_i.get("rms", np.nan)
                diag["sigma0"][i] = diag_i.get("sigma0", np.nan)
                diag["pred_se_center"][i] = diag_i.get("pred_se_center", np.nan)
                se_g = diag_i.get("se_grad", (np.nan, np.nan))
                diag["se_grad"][i, 0] = se_g[0]
                diag["se_grad"][i, 1] = se_g[1]
                diag["n_points"][i] = diag_i.get("n_points", 0)
                diag["cond"][i] = diag_i.get("cond", np.nan)
                diag["ok"][i] = True

        except Exception:
            # 失败时：保持 NaN/默认值，同时记录有效观测数，便于排查
            m = np.isfinite(V_block).sum()
            if return_diagnostics:
                diag["n_points"][i] = int(m)
                diag["ok"][i] = False
            continue

    return betas, grads, diag

def _wendland_c2(q):
    # φ(q) = (1-q)^4_+ * (1+4q)
    out = np.zeros_like(q, float)
    m = (q >= 0) & (q <= 1)
    t = 1.0 - q[m]
    out[m] = (t**4) * (1.0 + 4.0*q[m])
    return out

def fit_rbf_field_wendland(
    V_block, LAT_block, LON_block, center_lat, center_lon,
    *, rho_scale=2.0, lambda_rbf=1e-2, scale_anisotropy=True
):
    """
    在 alt-IPP 为中心，用 Wendland-C2 RBF(+线性尾项) 拟合 3×5/5×3 GIM 小块。
    返回 model，可对该块范围内任意点 f(x,y) 评估 VTEC。
    """
    V = np.asarray(V_block, float).ravel()
    La = np.asarray(LAT_block, float).ravel()
    Lo = np.asarray(LON_block, float).ravel()
    m = np.isfinite(V) & np.isfinite(La) & np.isfinite(Lo)
    if m.sum() < 6:
        return None

    Vi = V[m]; lat_i = La[m]; lon_i = Lo[m]
    xi, yi = _xy_km(center_lat, center_lon, lat_i, lon_i)

    # 各向异性尺度归一化（强烈建议开）
    sx = np.std(xi) if scale_anisotropy else 1.0
    sy = np.std(yi) if scale_anisotropy else 1.0
    sx = sx if sx > 0 else 1.0
    sy = sy if sy > 0 else 1.0
    xs, ys = xi/sx, yi/sy

    # 支撑半径 ρ：用点云尺度的 rho_scale 倍（可按需调大/调小）
    rs = np.sqrt(xs**2 + ys**2)
    base = max(np.std(xs), np.std(ys), 1.0)
    rho = rho_scale * base

    # RBF 矩阵 Φ_ij = φ(||ri-rj||/ρ)
    dist = np.sqrt((xs[:, None]-xs[None, :])**2 + (ys[:, None]-ys[None, :])**2)
    Phi = _wendland_c2(dist / rho)

    # 线性尾项 P = [1, x, y]
    P = np.column_stack([np.ones_like(xs), xs, ys])

    # 正规方程（RBF+线性尾项；λ 只正则化 RBF 部分）
    A = np.block([
        [Phi + lambda_rbf*np.eye(len(Vi)), P],
        [P.T, np.zeros((3, 3))]
    ])
    b = np.concatenate([Vi, np.zeros(3)])

    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]

    alpha = sol[:len(Vi)]
    beta  = sol[len(Vi):]  # a0, ax, ay

    # 训练残差与诊断
    Vhat = Phi @ alpha + P @ beta
    res = Vi - Vhat
    diag = {
        "rms": float(np.sqrt(np.mean(res**2))),
        "n_points": int(len(Vi)),
        "cond": float(np.linalg.cond(A)),
        "alpha_norm": float(np.linalg.norm(alpha)),
        "lat_min": float(np.nanmin(LAT_block)),
        "lat_max": float(np.nanmax(LAT_block)),
        "lon_min": float(np.nanmin(LON_block)),
        "lon_max": float(np.nanmax(LON_block)),
        "rho": float(rho), "sx": float(sx), "sy": float(sy),
        "lambda_rbf": float(lambda_rbf)
    }

    return {
        "alpha": alpha, "beta": beta,
        "xs": xs, "ys": ys,   # 样本的归一化坐标
        "sx": sx, "sy": sy,   # 还原梯度/坐标用
        "lat_c": float(center_lat), "lon_c": float(center_lon),
        "rho": float(rho),
        "diag": diag
    }

def eval_rbf_field_wendland(model, lat_q, lon_q):
    """在已拟合 RBF 模型上评估任意点（例如 DORIS IPP）的 VTEC 估计。"""
    if model is None:
        return np.nan
    alpha = model["alpha"]; beta = model["beta"]
    sx, sy = model["sx"], model["sy"]
    lat_c, lon_c = model["lat_c"], model["lon_c"]
    rho = model["rho"]
    xs = model["xs"]; ys = model["ys"]

    xq, yq = _xy_km(lat_c, lon_c, np.atleast_1d(lat_q), np.atleast_1d(lon_q))
    xqs, yqs = xq/sx, yq/sy

    dist = np.sqrt((xs[None, :] - xqs[:, None])**2 + (ys[None, :] - yqs[:, None])**2)
    Phi_q = _wendland_c2(dist / rho)            # (Q,P)
    P_q   = np.column_stack([np.ones_like(xqs), xqs, yqs])  # (Q,3)
    vq = (Phi_q @ alpha) + (P_q @ beta)
    return vq if vq.size > 1 else float(vq[0])