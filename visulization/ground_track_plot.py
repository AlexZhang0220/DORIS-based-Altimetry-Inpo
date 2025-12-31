import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import ast   # 用来安全地解析字符串为 list 对象
from pathlib import Path
def parse_str_list(s):
    """把字符串形式的 list 转为真正的 Python list[float]"""
    if isinstance(s, str):
        try:
            val = ast.literal_eval(s)  # 安全地把 '[-67.1, -67.2]' → [-67.1, -67.2]
            if isinstance(val, (list, tuple, np.ndarray)):
                return np.asarray(val, float)
            else:
                return np.array([], float)
        except Exception:
            return np.array([], float)
    elif isinstance(s, (list, np.ndarray)):
        return np.asarray(s, float)
    else:
        return np.array([], float)
    
start_epoch = pd.Timestamp("2016-03-12 22:51:41") 
end_epoch = pd.Timestamp("2016-03-12 23:47:53")

satellite = 'ja3'
year = start_epoch.year
doy = start_epoch.dayofyear

doris_path = f"/Volumes/T7Shield/DORISData/DORISObsStorage/{satellite}/{year}/DOY{doy:03d}.parquet"
alt_path   = f"/Volumes/T7Shield/AltimetryData/AltimetryVTECStorage/{satellite}/{year}/DOY{doy:03d}.pkl"
inpo_path = f"./visulization/slope test/OnlyDorisFieldNoWeight.pkl"
inpo_path_2 = f"./visulization/slope test/OnlyDorisFieldWeight.pkl"
inpo_path_3 = f"./visulization/slope test/OnlyDorisFieldWeightResudialEst.pkl"

# 读取数据
df_obs = pd.read_pickle(inpo_path)
df_obs_2 = pd.read_pickle(inpo_path_2)
df_obs_3 = pd.read_pickle(inpo_path_3)

data_dir = Path("/Volumes/T7Shield/DORISData/DBAVStorage/ja3/2000km/2024/quad_ratio")

df_list = []
for file in sorted(data_dir.glob("*.pkl")):
    if file.name.startswith("._"):
        continue  # 跳过系统文件
    df_tmp = pd.read_pickle(file)
    df_list.append(df_tmp)
    print(f"Loaded {file.name}, shape={df_tmp.shape}")

if df_list:
    df_obs_all = pd.concat(df_list, ignore_index=True)
# df_obs_all = df_list[0]
df_obs_selected_ratio = df_obs_all[df_obs_all['station_count'] > 1].reset_index(drop=True)
df_obs_low_ratio      = df_obs_all[df_obs_all['station_count'] == 1].reset_index(drop=True)
# --- 计算 RMS 函数 ---
def rms(series):
    s = series.dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(s**2))) if s.size else np.nan

# 计算四个数据集的差值序列
diff_sel_doris = df_obs_selected_ratio['doris_vtec'] - df_obs_selected_ratio['VTEC']
diff_sel_gim   = df_obs_selected_ratio['gim_vtec'] * 0.925 - df_obs_selected_ratio['VTEC']

diff_low_doris = df_obs_low_ratio['doris_vtec'] - df_obs_low_ratio['VTEC']
diff_low_gim   = df_obs_low_ratio['gim_vtec'] * 0.925 - df_obs_low_ratio['VTEC']

# --- 计算 RMS 和 STD ---
rms_vals = [
    rms(diff_sel_doris),
    rms(diff_sel_gim),
    rms(diff_low_doris),
    rms(diff_low_gim)
]

std_vals = [
    np.std(diff_sel_doris),
    np.std(diff_sel_gim),
    np.std(diff_low_doris),
    np.std(diff_low_gim)
]

labels = [
    "Selected DORIS-Alt",
    "Selected GIM*0.925-Alt",
    "Low DORIS-Alt",
    "Low GIM*0.925-Alt"
]

# --- 画图 ---
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# 坐标位置：左组 (Selected) 和右组 (Low)
labels = ["DORIS-Alt", "GIM*0.925-Alt"]
x_sel = np.arange(len(labels))
x_low = x_sel + len(labels) + 0.8  # 右移一点作为第二组

# --- 上面 RMS ---
axes[0].bar(x_sel, [rms_vals[0], rms_vals[1]], width=0.6, label="Selected")
axes[0].bar(x_low, [rms_vals[2], rms_vals[3]], width=0.6, label="Low")

axes[0].set_ylabel("RMS (TECU)")
axes[0].set_title("RMS Comparison")
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

# 标注数值
for x, v in zip(np.r_[x_sel, x_low],rms_vals):
    axes[0].text(x, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

# --- 下面 STD ---
axes[1].bar(x_sel, [std_vals[0], std_vals[1]], width=0.6, label="Selected")
axes[1].bar(x_low, [std_vals[2], std_vals[3]], width=0.6, label="Low")

axes[1].set_ylabel("STD (TECU)")
axes[1].set_title("STD Comparison")
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

for x, v in zip(np.r_[x_sel, x_low],std_vals):
    axes[1].text(x, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=9)

# --- 统一设置 ---
xticks = np.concatenate([x_sel, x_low])
xticklabels = labels + labels
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(xticklabels, rotation=0)

# 添加分组标签
axes[1].text(np.mean(x_sel), -0.08*max(std_vals),
             "Selected", ha='center', va='top', transform=axes[1].get_xaxis_transform())
axes[1].text(np.mean(x_low), -0.08*max(std_vals),
             "Low", ha='center', va='top', transform=axes[1].get_xaxis_transform())

plt.tight_layout()
plt.show()


## plot ground track of high sided ratio, demo
lat_arrays = df_obs_4['doris_lat'].apply(parse_str_list)
lon_arrays = df_obs_4['doris_lon'].apply(parse_str_list)
pairs = []
for lat_arr, lon_arr in zip(lat_arrays, lon_arrays):
    n = min(len(lat_arr), len(lon_arr))
    if n > 0:
        pairs.extend(zip(lat_arr[:n], lon_arr[:n]))
coords = np.unique(np.round(np.array(pairs, float), 6), axis=0)
unique_lat = coords[:, 0]
unique_lon = coords[:, 1]

lat_arrays = df_obs_selected_ratio['doris_lat'].apply(parse_str_list)
lon_arrays = df_obs_selected_ratio['doris_lon'].apply(parse_str_list)
pairs = []
for lat_arr, lon_arr in zip(lat_arrays, lon_arrays):
    n = min(len(lat_arr), len(lon_arr))
    if n > 0:
        pairs.extend(zip(lat_arr[:n], lon_arr[:n]))
coords = np.unique(np.round(np.array(pairs, float), 6), axis=0)
unique_lat_ratio = coords[:, 0]
unique_lon_ratio = coords[:, 1]

plt.scatter((df_obs_4['ipp_lon'] + 180.0) % 360.0 - 180.0, df_obs_4['ipp_lat'], s=5, label='all points', alpha=0.5)
plt.scatter((df_obs_selected_ratio['ipp_lon'] + 180.0) % 360.0 - 180.0, df_obs_selected_ratio['ipp_lat'], s=5, label='sided ratio >= 0.3', alpha=0.5)

plt.scatter(unique_lon, unique_lat, s=1, color='red', label='unique doris points')
plt.scatter(unique_lon_ratio, unique_lat_ratio, s=1, color='green', label='unique doris points sided ratio >= 0.3')
# 创建两个子图（上下排列）
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# -------- 子图1 --------
axes[0].scatter(df_obs['obs_epoch'], df_obs['doris_vtec'], s=10, label='DORIS VTEC', alpha=0.7)
axes[0].scatter(df_obs['obs_epoch'], df_obs['gim_vtec'] * 0.925, s=10, label='GIM VTEC × 0.925', alpha=0.7)
axes[0].scatter(df_obs['obs_epoch'], df_obs['VTEC'], s=10, label='Altimetry VTEC', alpha=0.7)
axes[0].set_xlabel('Observation Time (UTC)')
axes[0].set_ylabel('VTEC (TECU)')
axes[0].set_title('OnlyDorisFieldNoWeight')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.4)

# -------- 子图2 --------
axes[1].scatter(df_obs_2['obs_epoch'], df_obs_2['doris_vtec'], s=10, label='DORIS VTEC', alpha=0.7)
axes[1].scatter(df_obs_2['obs_epoch'], df_obs_2['gim_vtec'] * 0.925, s=10, label='GIM VTEC × 0.925', alpha=0.7)
axes[1].scatter(df_obs_2['obs_epoch'], df_obs_2['VTEC'], s=10, label='Altimetry VTEC', alpha=0.7)

axes[1].set_xlabel('Observation Time (UTC)')
axes[1].set_ylabel('VTEC (TECU)')
axes[1].set_title('OnlyDorisFieldWeight')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.4)

# -------- 子图3 --------
axes[2].scatter(df_obs_3['obs_epoch'], df_obs_3['doris_vtec'], s=10, label='DORIS VTEC', alpha=0.7)
axes[2].scatter(df_obs_3['obs_epoch'], df_obs_3['gim_vtec'] * 0.925, s=10, label='GIM VTEC × 0.925', alpha=0.7)
axes[2].scatter(df_obs_3['obs_epoch'], df_obs_3['VTEC'], s=10, label='Altimetry VTEC', alpha=0.7)

axes[2].set_xlabel('Observation Time (UTC)')
axes[2].set_ylabel('VTEC (TECU)')
axes[2].set_title('OnlyDorisFieldWeightResudialEst')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.4)

# 调整布局
plt.tight_layout()
plt.show()

df_doris = pd.read_parquet(doris_path)
df_altimetry = pd.read_pickle(alt_path)

df_doris_filtered = df_doris[
    (df_doris['obs_epoch'] >= start_epoch - pd.Timedelta(seconds=400)) &
    (df_doris['obs_epoch'] <= end_epoch + pd.Timedelta(seconds=400))
]      
df_altimetry_filtered = df_altimetry[
    (df_altimetry['obs_epoch'] >= start_epoch) &
    (df_altimetry['obs_epoch'] <= end_epoch)        
]

lon1 = df_doris_filtered['ipp_lon'].to_numpy()
lat1 = df_doris_filtered['ipp_lat'].to_numpy()
lon2 = df_altimetry_filtered['ipp_lon'].to_numpy()
lat2 = df_altimetry_filtered['ipp_lat'].to_numpy()

# —— 画图 —— #
# 假设 lon2, lat2 是 numpy 数组或列表
mid = len(lon2) // 2
lon2_first, lat2_first = lon2[:mid], lat2[:mid]
lon2_second, lat2_second = lon2[mid:], lat2[mid:]

mid = len(lon1) // 2
lon1_first, lat1_first = lon1[:mid], lat1[:mid]
lon1_second, lat1_second = lon1[mid:], lat1[mid:]

fig, axes = plt.subplots(
    2, 1, 
    subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)}
)

for i, (ax, lons, lats, lons_d, lats_d, title_suffix) in enumerate(zip(
    axes,
    [lon2_first, lon2_second],
    [lat2_first, lat2_second],
    [lon1_first, lon1_second],
    [lat1_first, lat1_second],
    ["(first half)", "(second half)"]
)):
    ax.set_global()
    ax.coastlines(linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="black")
    ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)

    ax.scatter(lons, lats, transform=ccrs.PlateCarree(), label="Altimetry")
    ax.scatter(lons_d, lats_d, transform=ccrs.PlateCarree(), label="DORIS", s=10, alpha=0.7)
    ax.legend(loc="lower left")
    ax.set_title(f"Ground Track in cycle3 pass121 {title_suffix}", fontsize=12)

plt.tight_layout()
plt.show()
