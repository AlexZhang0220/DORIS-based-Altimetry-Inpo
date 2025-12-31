import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.prepared import prep
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# =========================
# Paths
# =========================
INPUT_PATH = r"/Users/alexzhang/Doctoral Program/DORIS-based Altimetry Inpo/visulization/doris-station-log.csv"
OUTPUT_PNG = r"doris_mean_distance.png"

# =========================
# DMS utilities
# =========================
def dms_to_decimal(dms_str):
    s = str(dms_str).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    sign = 1.0
    s_up = s.upper()
    if "S" in s_up or "W" in s_up:
        sign = -1.0
    if s.startswith("-"):
        sign = -1.0

    for ch in ["°", "º", "'", "’", '"', "″", ":", ","]:
        s = s.replace(ch, " ")
    s = s.replace("N", "").replace("E", "").replace("S", "").replace("W", "")

    parts = [p for p in s.split() if p]
    if len(parts) == 1:
        return sign * float(parts[0])

    deg = float(parts[0])
    minute = float(parts[1]) if len(parts) > 1 else 0.0
    sec = float(parts[2]) if len(parts) > 2 else 0.0
    return sign * (abs(deg) + minute / 60 + sec / 3600)

def wrap_lon(lon):
    return (lon + 180) % 360 - 180

# =========================
# Great-circle distance
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# =========================
# Mean distance grid
# =========================
def compute_mean2nn_grid(sta_lat, sta_lon, lat_lim=66.5, ddeg=1.0):
    lats = np.arange(-lat_lim, lat_lim + 1e-9, ddeg)
    lons = np.arange(-180, 180 + 1e-9, ddeg)
    lon2d, lat2d = np.meshgrid(lons, lats)

    dist = haversine_km(
        lat2d[..., None], lon2d[..., None],
        sta_lat[None, None, :], sta_lon[None, None, :]
    )

    idx2 = np.argpartition(dist, 1, axis=2)[..., :2]
    d2 = np.take_along_axis(dist, idx2, axis=2)
    return lon2d, lat2d, d2.mean(axis=2)

# =========================
# Land mask (ocean only)
# =========================
def mask_land(lon2d, lat2d):
    land = cfeature.NaturalEarthFeature(
        "physical", "land", "110m"
    )
    geom = prep(next(iter(land.geometries())))

    mask = np.zeros(lon2d.shape, dtype=bool)
    for i in range(lon2d.shape[0]):
        for j in range(lon2d.shape[1]):
            mask[i, j] = geom.contains(Point(lon2d[i, j], lat2d[i, j]))
    return mask

# =========================
# Main
# =========================
def main():
    # ---- read stations ----
    df = pd.read_csv(INPUT_PATH)

    # cols_xyz = ["sta_x", "sta_y", "sta_z"]
    # df_clean = df.dropna(subset=cols_xyz).copy()
    #     # WGS84 constants
    # a = 6378137.0
    # f = 1 / 298.257223563
    # e2 = f * (2 - f)

    # x = df_clean["sta_x"].to_numpy()
    # y = df_clean["sta_y"].to_numpy()
    # z = df_clean["sta_z"].to_numpy()

    # lon = np.arctan2(y, x)

    # p = np.sqrt(x**2 + y**2)
    # lat = np.arctan2(z, p * (1 - e2))

    # # one iteration is enough for station-level accuracy
    # for _ in range(2):
    #     sin_lat = np.sin(lat)
    #     N = a / np.sqrt(1 - e2 * sin_lat**2)
    #     lat = np.arctan2(z + e2 * N * sin_lat, p)

    # sta_lon = np.degrees(lon)
    # sta_lat = np.degrees(lat)
    df.columns = [c.lower().strip() for c in df.columns]
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    df["station"] = df["station"].str.strip()
    df["status"] = np.where(df["station"].str.endswith("*"), "inactive", "active")
    df["station"] = df["station"].str.rstrip("*")

    df["lat"] = df["latitude"].apply(dms_to_decimal)
    df["lon"] = df["longitude"].apply(dms_to_decimal).apply(wrap_lon)
    df = df.dropna(subset=["lat", "lon"])

    df_active = df[df["status"] == "active"]
    sta_lat = df_active["lat"].astype(float).values
    sta_lon = df_active["lon"].astype(float).values

    # ---- compute distance field ----
    lon2d, lat2d, mean_dist = compute_mean2nn_grid(sta_lat, sta_lon, ddeg=1.0)

    # ---- mask land ----
    land_mask = mask_land(lon2d, lat2d)
    mean_dist = np.where(land_mask, np.nan, mean_dist)
    # mean_dist_cut = np.where(mean_dist <= 1675, mean_dist, np.nan)

    # ===== Global style =====
    plt.rcParams.update({
        "font.family": "Arial",

        # global
        "font.size": 10,

        # axis labels
        "axes.labelsize": 12,
        "axes.titlesize": 12,

        # ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # legend
        "legend.fontsize": 10,

        # axis spine
        "axes.linewidth": 1.0,
    })
    fig, ax = plt.subplots(
        figsize=(7, 3.5),   # JGR single-column width
        subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    ax.set_extent([-180, 180, -66.5, 66.5], crs=ccrs.PlateCarree())

    # ---- Distance field ----
    im = ax.pcolormesh(
        lon2d, lat2d, mean_dist,
        cmap="viridis",          # JGR-friendly colormap
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    # ---- Land & coastlines ----
    ax.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=3)
    ax.coastlines(linewidth=0.6, zorder=4)

    # ---- Stations ----
    ax.scatter(
        sta_lon, sta_lat,
        s=35,
        facecolors="#e53935",
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
        transform=ccrs.PlateCarree()
    )

    # ===== Gridlines + labels =====
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        color="0.4",
        alpha=0.6,
        linestyle="--",
        zorder=2
    )

    gl.top_labels = False
    gl.right_labels = False

    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 61, 30))

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # ===== Colorbar (bottom) =====
    cbar = plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.08,        # 与主图距离
        fraction=0.06    # 色条高度
    )

    cbar.set_label(
        "Mean distance to nearest two DORIS stations (km)",
        family="Arial"
    )

    # ---- No title by design ----

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()