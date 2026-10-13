# DORIS VTEC and Altimetry Interpolation

Derive vertical total electron content (VTEC) from dual-frequency DORIS
observations, then interpolate it onto satellite altimetry ionospheric
pierce points (IPPs). The repository retains the original two-stage DORIS
workflow and a separate cycle-level altimetry interpolation workflow.

```text
DORIS RINEX + SP3 + SINEX
          │
          ▼
Stage 1: geometry and raw STEC ──► daily .pkl
          │
          ▼  UTC GIM reference
Stage 2: arc detection and leveling ──► daily VTEC .pickle
          │
          ▼  altimetry IPPs + cycle tables + GIM + SINEX
INPO: DORIS-to-altimetry interpolation ──► cycleNNN.pickle
```

## Setup

Use a Python environment with these runtime packages:
```bash
python -m pip install numpy pandas scipy pyproj georinex xarray numba tqdm astropy
```

Astropy provides the historical TAI/UTC conversion; it is required by the
corrected workflow.
This repository does not download input data. Prepare DORIS RINEX, SP3,
SINEX and supported IGS IONEX files. INPO additionally requires daily
altimetry products and cycle-definition CSV files.

## Configure and run

Run the modules from the repository root. They use parameters defined in
their source files; there is no command-line date or stage selector.

1. Edit [`doris_vtec/main.py`](doris_vtec/main.py): set the satellite,
   `start_date`, `proc_days`, `block_days`, and all input/output roots.
   Defaults select Jason-3 (`ja3`) and 11 days starting 2018-01-01.
2. Review the processing settings in
   [`doris_vtec/config.py`](doris_vtec/config.py), then run both DORIS stages:

   ```bash
   python -m doris_vtec.main
   ```

3. Edit [`doris_inpo/main.py`](doris_inpo/main.py): set the satellite,
   `cycle_list`, data roots, cycle-table paths and worker count. Defaults
   select Jason-3 cycle 70. Review
   [`doris_inpo/config.py`](doris_inpo/config.py), then run:

   ```bash
   python -m doris_inpo.main
   ```

Change the example absolute data paths for your installation. Use new output directories for
validation: executing a pipeline can replace an existing product at the same
path. The satellite list in the example driver is not a guarantee of full
support; the retained detector is specific to Jason-3 (`L39`) with supported
STAREC/ALCATEL antennas.

## Products and interfaces

| Product | Path below its configured root | Content |
| --- | --- | --- |
| Stage 1 | `YEAR/DOYddd.pkl` | Observations, corrected time, geometry and raw STEC |
| Stage 2 | `YEAR/DOYddd.pickle` | Leveled DORIS STEC and VTEC |
| INPO | `cycle_unit/cycleNNN.pickle` | Altimetry rows with interpolated DORIS and reference GIM VTEC |

Products are pandas DataFrames readable with `pandas.read_pickle`. The
Stage 2 public schema remains these ten columns:

```text
obs_epoch, ipp_lat, ipp_lon, stec, vtec, elevation,
map_value, station_code, prn, pass_id
```

Here `stec` is the leveled STEC, not the raw Stage 1 value. Product identity
is also stored in DataFrame `attrs`; preserve these attributes when passing
products between stages.

## Fixes

- Correct endpoint LOS projections and separate antenna PCO constants.
- Read complete RINEX receiver-clock fields and apply observation scaling.
- Align observation times with GIM and SINEX queries.
- Handle missing phases, conflicting orbit records and station solution changes.
- Handle GIM missing values, UTC day boundaries and gaps between maps.
- Restore cycle-edge support, zero-distance IDW and the INPO output path.
