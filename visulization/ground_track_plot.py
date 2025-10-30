import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

start_epoch = pd.Timestamp("2016-03-12 22:51:41") 
end_epoch = pd.Timestamp("2016-12-31 23:47:53")

satellite = 'ja3'
year = start_epoch.year
doy = start_epoch.dayofyear

doris_path = f"/Volumes/T7Shield/DORISData/DORISObsStorage/{satellite}/{year}/DOY{doy:03d}.parquet"
alt_path   = f"/Volumes/T7Shield/AltimetryData/AltimetryVTECStorage/{satellite}/{year}/DOY{doy:03d}.pkl"
inpo_path = f"./visulization/slope test/OnlyDorisFieldNoWeight.pkl"

inpo_path_2 = f"./visulization/slope test/OnlyDorisFieldWeight.pkl"

# 读取数据
df_obs = pd.read_pickle(inpo_path)
df_obs_2 = pd.read_pickle(inpo_path_2)

# 创建两个子图（上下排列）
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# -------- 子图1 --------
axes[0].scatter(df_obs['obs_epoch'], df_obs['doris_vtec'], s=10, label='DORIS VTEC', alpha=0.7)
axes[0].scatter(df_obs['obs_epoch'], df_obs['gim_vtec'] * 0.925, s=10, label='GIM VTEC × 0.925', alpha=0.7)
axes[0].scatter(df_obs['obs_epoch'], df_obs['VTEC'], s=10, label='Altimetry VTEC', alpha=0.7)

axes[0].set_ylabel('VTEC (TECU)')
axes[0].set_title('Comparison of VTEC OnlyDorisFieldNoWeight')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.4)

# -------- 子图2 --------
axes[1].scatter(df_obs_2['obs_epoch'], df_obs_2['doris_vtec'], s=10, label='DORIS VTEC', alpha=0.7)
axes[1].scatter(df_obs_2['obs_epoch'], df_obs_2['gim_vtec'] * 0.925, s=10, label='GIM VTEC × 0.925', alpha=0.7)
axes[1].scatter(df_obs_2['obs_epoch'], df_obs_2['VTEC'], s=10, label='Altimetry VTEC', alpha=0.7)

axes[1].set_xlabel('Observation Time (UTC)')
axes[1].set_ylabel('VTEC (TECU)')
axes[1].set_title('Comparison of VTEC OnlyDorisFieldWeight')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.4)

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
