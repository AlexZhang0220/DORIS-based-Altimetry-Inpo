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
inpo_path = f"./visulization/slope test/cycle3pass121.pkl"

df_obs = pd.read_pickle(inpo_path)
plt.figure(figsize=(10, 5))
plt.scatter(df_obs['obs_epoch'], df_obs['doris_vtec'], s=10, label='DORIS VTEC', alpha=0.7)
plt.scatter(df_obs['obs_epoch'], df_obs['gim_vtec'] * 0.925,   s=10, label='GIM VTEC',   alpha=0.7)
plt.scatter(df_obs['obs_epoch'], df_obs['VTEC'],       s=10, label='Altimetry VTEC', alpha=0.7)

plt.xlabel('Observation Time (UTC)')
plt.ylabel('VTEC (TECU)')
plt.title('Comparison of VTEC Sources')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
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
fig = plt.figure(figsize=(11, 5.5))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))  # 0°为中心
ax.set_global()
ax.coastlines(linewidth=1.0)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="black")
ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
ax.scatter(lon1, lat1, transform=ccrs.PlateCarree(), label="DORIS")
ax.scatter(lon2, lat2, transform=ccrs.PlateCarree(), label="Altimetry")

ax.legend(loc="lower left")
ax.set_title("Ground Track in cycle3 pass121", fontsize=12)
plt.tight_layout()
plt.show()

