import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D

def plot_doris_map(
    df,
    lon_col="lon", lat_col="lat", status_col="status",
    projection=ccrs.Robinson(),
    figsize=(14, 6),
    land_color="0.75",           # 灰色陆地
    ocean_color="white",         # 白色海洋
    gridline_color="0.55",
    active_color="#e53935",      # 红色
    dot_size=70,
    savepath=None, dpi=300
):
    """
    df: 包含经纬度与状态的 DataFrame。status 取值: 'active' or 'inactive'
    """
    # 投影与画布
    proj = projection
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.set_global()

    # 海陆要素
    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=land_color, edgecolor="none")
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=ocean_color, edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.5, color="black")
    ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.4, color="black")

    # 经纬网
    gl = ax.gridlines(draw_labels=False, linewidth=0.5, color=gridline_color, alpha=0.7, linestyle='-')
    # Robinson 投影下，标注经纬度一般不加标签更干净；需要可改 draw_labels=True

    # 画站点
    geo = ccrs.Geodetic()  # 数据坐标系（经纬度）
    df_active = df[df[status_col].str.lower()=="active"]
    # df_inactive = df[df[status_col].str.lower()!="active"]

    # active: 实心红点
    ax.scatter(
        df_active[lon_col].values, df_active[lat_col].values,
        transform=geo, s=dot_size, zorder=5,
        facecolor=active_color, edgecolor="black", linewidth=0.6
    )

    # 如果你的 Cartopy 比较新，也可以使用：
    ax.spines['geo'].set_linewidth(1.5)
    ax.spines['geo'].set_edgecolor("black")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()


def dms_to_decimal(s):
    if pd.isna(s) or not str(s).strip(): return np.nan
    s = str(s).strip()
    try: return float(s)
    except ValueError: pass
    sign = -1 if s.startswith('-') else 1
    s = s.lstrip('+-')
    for ch in ['°', "'", '"', '′', '″', '’', '”', ':', 'm']:
        s = s.replace(ch, ' ')
    parts = [p for p in s.split() if p]
    deg = float(parts[0]); minutes = float(parts[1]) if len(parts)>1 else 0.0; seconds = float(parts[2]) if len(parts)>2 else 0.0
    return sign * (deg + minutes/60 + seconds/3600)

def wrap_lon(lon):
    if lon > 180: lon -= 360
    if lon < -180: lon += 360
    return lon


INPUT_PATH = r"./visulization/doris-station-log.csv"
OUTPUT_PNG = r"./visulization/doris_stations_robinson.png"

df = pd.read_csv(INPUT_PATH, sep=None, engine="python", encoding="utf-8-sig", dtype=str)
df.columns = [c.strip().lower() for c in df.columns]
df["status"] = 'active'
for i in range(len(df)):
    if df['station'][i][-1] == '*':
        df["status"][i] = 'inactive'
df["lat"] = df["latitude"].apply(dms_to_decimal)
df["lon"] = df["longitude"].apply(dms_to_decimal).apply(wrap_lon)
df = df.dropna(subset=["lat","lon"])

plot_doris_map(df, savepath=OUTPUT_PNG)