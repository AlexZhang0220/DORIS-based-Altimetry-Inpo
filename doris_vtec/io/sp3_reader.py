import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Iterable

from georinex import load as load_sp3


# ============================================================
# Internal helpers
# ============================================================

def _load_sp3_datasets(sp3_files: Iterable[Path]) -> xr.Dataset:
    """
    Load and concatenate SP3 files into one xarray Dataset.
    """
    ds_list = [load_sp3(str(f)) for f in sp3_files]
    return xr.concat(ds_list, dim="time")


def _pivot_sp3(ds: xr.Dataset, name: str) -> pd.DataFrame:
    """Deduplicate matching records without averaging conflicting states."""
    keys = ["time", "sv", "ECEF"]
    rows = ds[name].to_dataframe().reset_index()[keys + [name]]
    rows = rows.dropna(subset=[name])
    if not np.isfinite(rows[name].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite SP3 {name} records.")
    if rows.groupby(keys)[name].nunique().gt(1).any():
        raise ValueError(f"Conflicting SP3 {name} records.")

    rows = rows.drop_duplicates(keys)
    return (
        rows.pivot(index=["time", "sv"], columns="ECEF", values=name)
        .reindex(columns=["x", "y", "z"])
        .reset_index()
    )


def _sp3_dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """
    Convert SP3 xarray Dataset to a pandas DataFrame.
    """
    # -------- position --------
    df_pos = (
        _pivot_sp3(ds, "position")
        .rename(columns={
            "time": "epoch",
            "sv": "prn",
        })
    )

    # Unit conversion: km -> m
    df_pos[["x", "y", "z"]] *= 1e3

    # -------- velocity --------
    df_vel = (
        _pivot_sp3(ds, "velocity")
        .rename(columns={
            "time": "epoch",
            "sv": "prn",
            "x": "vx",
            "y": "vy",
            "z": "vz",
        })
    )

    # -------- merge --------
    df = pd.merge(
        df_pos,
        df_vel,
        on=["epoch", "prn"],
        how="left",
    )

    return df.sort_values("epoch").reset_index(drop=True)


# ============================================================
# Public API 
# ============================================================

def read_sp3_files(sp3_files: Iterable[Path]) -> pd.DataFrame:
    """
    Read SP3 orbit files and return satellite state vectors.

    Units:
        - x, y, z   : meters (m)
        - vx, vy, vz: meters per second (m/s)

    Returns
    -------
    pd.DataFrame
        Columns:
        - epoch
        - prn
        - x, y, z        (m)
        - vx, vy, vz     (m/s)
    """
    sp3_files = list(sp3_files)
    if not sp3_files:
        raise ValueError("No SP3 files provided.")

    ds = _load_sp3_datasets(sp3_files)
    return _sp3_dataset_to_dataframe(ds)
